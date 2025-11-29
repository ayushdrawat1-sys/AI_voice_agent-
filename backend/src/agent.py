# agent.py
import logging
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging & env
# -------------------------
logger = logging.getLogger("game_master_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# load .env.local in the backend folder (create/edit as needed)
load_dotenv(".env.local")

# -------------------------
# Configurable voice IDs & options (set these in .env.local)
# -------------------------
# If you don't have a specific voice id, leave AUREK_VOICE as a placeholder;
# SSML prosody will still attempt to deepen the voice.
AUREK_VOICE = os.getenv("AUREK_VOICE", "en-GB-deep-01")
KEN_VOICE = os.getenv("KEN_VOICE", "ken")  # kept for compatibility if Ken used
TTS_USE_SSML = os.getenv("TTS_USE_SSML", "true").lower() in ("1", "true", "yes")
TTS_WAIT_FOR_PLAYBACK = os.getenv("TTS_WAIT_FOR_PLAYBACK", "true").lower() in ("1", "true", "yes")

# -------------------------
# Local helpers
# -------------------------
BASE_DIR = Path(__file__).parent.resolve()

# -------------------------
# Witcher-style World (mini-arc) (Aurek + optional Ken integration)
# -------------------------
WORLD = {
    "intro": {
        "title": "A Shadow over Ard Skellig",
        "desc": (
            "You wake on the black-sand shore of a ravaged isle. Weathered posts creak in the wind; "
            "a half-sunk galley lies further down. Above, the ruined keep of Keldmar watches the coast. "
            "A stranger's silvered coin gleams near your boot, and the scent of brine carries old blood."
        ),
        "choices": {
            "inspect_coin": {"desc": "Inspect the silvered coin.", "result_scene": "coin"},
            "head_to_keep": {"desc": "Make for the ruined keep.", "result_scene": "keep_approach"},
            "follow_smoke": {"desc": "Follow smoke in the distance.", "result_scene": "village"},
        },
    },
    "coin": {
        "title": "The Silvered Coin",
        "desc": (
            "The coin bears a witcher-like sigil scratched into it. As you rub it clean, "
            "a low hum brushes your skin — the kind monsters feel near."
        ),
        "choices": {
            "take_coin": {
                "desc": "Pocket the coin (it tugs at you).",
                "result_scene": "keep_approach",
                "effects": {"add_inventory": "silvered_coin", "add_journal": "Found coin with a witcher-like sigil."},
            },
            "leave_coin": {"desc": "Leave the coin be.", "result_scene": "intro"},
        },
    },
    "keep_approach": {
        "title": "Keldmar Keep",
        "desc": (
            "The keep's gate hangs open. Inside, flickers of movement and a bitter, metallic smell. "
            "A lone torch lights a stairwell spiraling down — or you can circle the perimeter."
        ),
        "choices": {
            "enter_gate": {"desc": "Step into the keep through the gate.", "result_scene": "inner_courtyard"},
            "circle_perimeter": {"desc": "Circle the perimeter for other entrances.", "result_scene": "secret_cellar"},
            "retreat": {"desc": "Return to the shore.", "result_scene": "intro"},
        },
    },
    "secret_cellar": {
        "title": "Hidden Cellar",
        "desc": (
            "A half-hidden trapdoor opens to a cellar lit by algae-lamps. On a table lies a rusted key and sealed scroll."
        ),
        "choices": {
            "take_key": {
                "desc": "Take the rusted key.",
                "result_scene": "cellar_key",
                "effects": {"add_inventory": "rusted_key", "add_journal": "Rusted key taken from cellar."},
            },
            "read_scroll": {
                "desc": "Break the seal and read the scroll.",
                "result_scene": "scroll_reveal",
                "effects": {"add_journal": "Scroll: 'The thing beneath remembers the tide.'"},
            },
            "leave": {"desc": "Back away quietly.", "result_scene": "keep_approach"},
        },
    },
    "inner_courtyard": {
        "title": "Courtyard at Dusk",
        "desc": (
            "A hunched, brine-streaked beast scuttles under the battlements. Its eyes flash. You have seconds."
        ),
        "choices": {
            "fight": {"desc": "Ready your blade and strike.", "result_scene": "fight_win"},
            "hide": {"desc": "Slip into shadow and observe.", "result_scene": "observe"},
            "flee": {"desc": "Flee back out to the shore.", "result_scene": "intro"},
        },
    },
    "cellar_key": {
        "title": "Key and Consequence",
        "desc": (
            "The key thrums faintly in your palm. From below, the tide sings a wrong note. A voice asks: 'Return it, witcher?'"
        ),
        "choices": {
            "pledge": {
                "desc": "Swear to return the lost thing.",
                "result_scene": "reward",
                "effects": {"add_journal": "Pledged to return the lost heirloom."},
            },
            "take": {
                "desc": "Pocket the key and leave.",
                "result_scene": "cursed_key",
                "effects": {"add_journal": "Pocketed the rusted key; a chill settles."},
            },
            "call_ken": {"desc": "Call for Ken the Wizard's advice (if you know him).", "result_scene": "ken_arrives"},
        },
    },
    "fight_win": {
        "title": "After the Scuffle",
        "desc": (
            "You fend off the creature. Among the wreckage is a carved locket with a family crest — perhaps the lost heirloom."
        ),
        "choices": {
            "take_locket": {
                "desc": "Take the locket.",
                "result_scene": "reward",
                "effects": {"add_inventory": "carved_locket", "add_journal": "Recovered carved locket."},
            },
            "leave_locket": {"desc": "Leave it and tend wounds.", "result_scene": "intro"},
        },
    },
    "reward": {
        "title": "A Witcher's Quiet",
        "desc": (
            "The night eases. The small arc of this tale closes; whether you return the heirloom or keep it will color more nights."
        ),
        "choices": {
            "end_session": {"desc": "End the session (conclude mini-arc).", "result_scene": "intro"},
            "explore_more": {"desc": "Keep searching the isle.", "result_scene": "intro"},
        },
    },
    "cursed_key": {
        "title": "Cold in the Palm",
        "desc": (
            "The key sits heavy. You feel a tug of sorrow and distant waves — some debts are deeper than coin."
        ),
        "choices": {
            "seek": {"desc": "Seek a way to undo the weight.", "result_scene": "reward"},
            "bury": {"desc": "Bury the key.", "result_scene": "intro"},
        },
    },
    "observe": {
        "title": "Hidden Observation",
        "desc": "From shadow you note the beast's manner: it flees from something deeper beneath the keep.",
        "choices": {"follow": {"desc": "Follow where it fled.", "result_scene": "secret_cellar"}, "retreat": {"desc": "Retreat.", "result_scene": "intro"}},
    },
    "village": {
        "title": "Smokecleft Village",
        "desc": "You find a small hamlet with shuttered doors and a single lamp in a window. A fisher mutters of tides and lost things.",
        "choices": {"ask_fisher": {"desc": "Talk to the fisher.", "result_scene": "fisher_talk"}, "return": {"desc": "Return to shore.", "result_scene": "intro"}},
    },
    "fisher_talk": {
        "title": "Fisher's Tale",
        "desc": "The fisher says: 'There is a thing beneath Keldmar. People don't speak of it.'",
        "choices": {"thank": {"desc": "Thank the fisher and move on.", "result_scene": "intro"}},
    },
    "ken_arrives": {
        "title": "Ken the Wizard",
        "desc": (
            "A streak of pale light and the scent of crushed rosemary heralds Ken's approach. "
            "Ken, a small wiry wizard with a crooked hat, peers at the key and scratches his chin."
        ),
        "choices": {
            "ask_ken": {"desc": "Ask Ken what he knows of the key.", "result_scene": "ken_speaks"},
            "dismiss_ken": {"desc": "Tell Ken to leave you be.", "result_scene": "cellar_key"},
        },
    },
    "ken_speaks": {
        "title": "Ken's Counsel",
        "desc": (
            "Ken cocks his head. 'Ah — a binding key. Offer contrition and the tide might yield what it keeps. "
            "But keep it, and you'll hear the sea at midnight.'"
        ),
        "choices": {
            "follow_ken": {"desc": "Take Ken's counsel and pledge to return the heirloom.", "result_scene": "reward", "effects": {"add_journal": "Heard Ken's counsel about the key."}},
            "ignore_ken": {"desc": "Ignore Ken and pocket the key.", "result_scene": "cursed_key", "effects": {"add_journal": "Ignored Ken and pocketed the key."}},
        },
    },
}

# -------------------------
# Per-session Userdata
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    current_scene: str = "intro"
    history: List[Dict] = field(default_factory=list)
    journal: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    named_npcs: Dict[str, str] = field(default_factory=dict)
    choices_made: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# -------------------------
# Helper functions for GM
# -------------------------
def scene_text(scene_key: str, userdata: Userdata) -> str:
    scene = WORLD.get(scene_key)
    if not scene:
        return "You are in a featureless void. What do you do?"

    desc = f"{scene.get('desc','')}\n\nChoices:\n"
    for cid, cmeta in scene.get("choices", {}).items():
        desc += f"- {cmeta.get('desc','')} (say: {cid})\n"
    desc += "\nWhat do you do?"
    return desc


def apply_effects(effects: dict, userdata: Userdata):
    if not effects:
        return
    if "add_journal" in effects:
        userdata.journal.append(effects["add_journal"])
    if "add_inventory" in effects:
        userdata.inventory.append(effects["add_inventory"])


def summarize_scene_transition(old_scene: str, action_key: str, result_scene: str, userdata: Userdata) -> str:
    entry = {
        "from": old_scene,
        "action": action_key,
        "to": result_scene,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    userdata.history.append(entry)
    userdata.choices_made.append(action_key)
    return f"You chose '{action_key}'."


# -------------------------
# Per-speaker SSML / voice helpers
# -------------------------
def get_voice_for_speaker(speaker: str) -> str:
    sp = (speaker or "").lower()
    if sp == "ken":
        return KEN_VOICE
    return AUREK_VOICE


def wrap_with_ssml(text: str, speaker: str = "aurek") -> str:
    """
    Wrap text in SSML with voice and prosody hints.
    If SSML disabled via env, returns plain text.
    """
    if not TTS_USE_SSML:
        return text

    voice_id = get_voice_for_speaker(speaker)
    if speaker.lower() == "ken":
        # Ken: slightly higher, whimsical cadence
        prosody = "<prosody pitch='+1st' rate='98%'>"
        break_time = "200ms"
    else:
        # Aurek: deeper, slower, more gruff
        # NOTE: -5st (semitones) and 92% rate for a natural deepening
        prosody = "<prosody pitch='-5st' rate='92%'>"
        break_time = "300ms"

    # Standard SSML with voice name wrapper
    ssml = f"<speak><voice name=\"{voice_id}\">{prosody}{text}</prosody></voice><break time='{break_time}'/></speak>"
    return ssml


# -------------------------
# GM Tools (function_tool)
# -------------------------
@function_tool
async def start_adventure(ctx: RunContext, player_name: Optional[str] = None):
    userdata: Userdata = ctx.userdata
    if player_name:
        userdata.player_name = player_name
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    opening = f"Greetings {userdata.player_name or 'traveler'}. Welcome to '{WORLD['intro']['title']}'.\n\n" + scene_text("intro", userdata)
    if not opening.endswith("What do you do?"):
        opening += "\nWhat do you do?"
    return wrap_with_ssml(opening, speaker="aurek")


@function_tool
async def get_scene(ctx: RunContext):
    userdata: Userdata = ctx.userdata
    scene_k = userdata.current_scene or "intro"
    return wrap_with_ssml(scene_text(scene_k, userdata), speaker="aurek")


@function_tool
async def player_action(ctx: RunContext, action: str):
    userdata: Userdata = ctx.userdata
    current = userdata.current_scene or "intro"
    scene = WORLD.get(current)
    action_text = (action or "").strip()

    chosen_key = None
    if scene and action_text.lower() in (scene.get("choices") or {}):
        chosen_key = action_text.lower()

    if not chosen_key and scene:
        for cid, cmeta in (scene.get("choices") or {}).items():
            desc = cmeta.get("desc", "").lower()
            if cid in action_text.lower() or any(w in action_text.lower() for w in desc.split()[:4]):
                chosen_key = cid
                break

    if not chosen_key and scene:
        for cid, cmeta in (scene.get("choices") or {}).items():
            for keyword in cmeta.get("desc", "").lower().split():
                if keyword and keyword in action_text.lower():
                    chosen_key = cid
                    break
            if chosen_key:
                break

    if not chosen_key:
        resp = (
            "I didn't quite catch that action for this situation. Try one of the listed choices or use a simple phrase like 'inspect the coin' or 'go to the keep'.\n\n"
            + scene_text(current, userdata)
        )
        return wrap_with_ssml(resp, speaker="aurek")

    choice_meta = scene["choices"].get(chosen_key)
    result_scene = choice_meta.get("result_scene", current)
    effects = choice_meta.get("effects", None)

    apply_effects(effects or {}, userdata)

    _note = summarize_scene_transition(current, chosen_key, result_scene, userdata)

    userdata.current_scene = result_scene

    # Choose speaker: scenes starting with 'ken_' use Ken, otherwise Aurek
    speaker = "aurek"
    if result_scene.startswith("ken_"):
        speaker = "ken"

    persona_tag = "Aurek (low, gravelly):\n\n" if speaker == "aurek" else "Ken (soft, curious):\n\n"
    next_desc = scene_text(result_scene, userdata)
    reply_text = f"{persona_tag}{_note}\n\n{next_desc}"
    if not reply_text.endswith("What do you do?"):
        reply_text += "\nWhat do you do?"
    return wrap_with_ssml(reply_text, speaker=speaker)


@function_tool
async def show_journal(ctx: RunContext):
    userdata: Userdata = ctx.userdata
    lines = []
    lines.append(f"Session: {userdata.session_id} | Started at: {userdata.started_at}")
    if userdata.player_name:
        lines.append(f"Player: {userdata.player_name}")
    if userdata.journal:
        lines.append("\nJournal entries:")
        for j in userdata.journal:
            lines.append(f"- {j}")
    else:
        lines.append("\nJournal is empty.")
    if userdata.inventory:
        lines.append("\nInventory:")
        for it in userdata.inventory:
            lines.append(f"- {it}")
    else:
        lines.append("\nNo items in inventory.")
    lines.append("\nRecent choices:")
    for h in userdata.history[-6:]:
        lines.append(f"- {h['time']} | from {h['from']} -> {h['to']} via {h['action']}")
    lines.append("\nWhat do you do?")
    return wrap_with_ssml("\n".join(lines), speaker="aurek")


@function_tool
async def restart_adventure(ctx: RunContext):
    userdata: Userdata = ctx.userdata
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"
    greeting = "The world resets. A new tide laps at the shore. You stand once more at the beginning.\n\n" + scene_text("intro", userdata)
    if not greeting.endswith("What do you do?"):
        greeting += "\nWhat do you do?"
    return wrap_with_ssml(greeting, speaker="aurek")


# -------------------------
# prewarm VAD (silero)
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


# -------------------------
# GameMaster Agent class
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        instructions = """
        You are 'Aurek', the Game Master (GM) for a voice-only, D&D-style short adventure set in a Witcher-inspired coastal region.
        Universe: Dark fantasy, monster-haunted isles and ruined keeps.
        Tone: Gruff, wry, cinematic — slightly world-weary but still tender. Do not imitate any real actor.
        Role: You are the GM. Describe scenes vividly, track inventory, NPCs, named locations, and always end your messages with 'What do you do?'
        Rules:
         - Use the provided tools (start_adventure, get_scene, player_action, show_journal, restart_adventure).
         - Keep continuity in userdata and reference journal/inventory when relevant.
         - Responses should be concise enough for spoken delivery but evocative.
         - Encourage choices; aim for ~8–15 exchanges and reach a mini-arc resolution.
         - When player input is ambiguous, ask a short clarifying question but still list the current choices.
        """
        super().__init__(
            instructions=instructions,
            tools=[start_adventure, get_scene, player_action, show_journal, restart_adventure],
        )


# -------------------------
# Entrypoint
# -------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🎲" * 6)
    logger.info("🚀 STARTING VOICE GAME MASTER (Ard Skellig mini-arc)")

    userdata = Userdata()

    # TTS kwargs
    tts_kwargs = dict(
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
    )
    if TTS_WAIT_FOR_PLAYBACK:
        # plugin-specific; safe to set if supported
        tts_kwargs["wait_for_playback"] = True

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=AUREK_VOICE,
            style="Conversation",
            **tts_kwargs,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=False,  # reduce partial/interrupted audio
        userdata=userdata,
    )

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
