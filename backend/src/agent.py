"""
Day 10 – Voice Improv Battle (Retro Arcade Host)
final day
Retro changes:
- Host persona updated to a retro/arcade-synthwave MC ("Neon MC") with matching intro language.
- Core behaviour and function interfaces unchanged (start_show, next_scenario, record_performance, summarize_show, stop_show).
- JSON export added: session data saved to ./sessions/<session_id>.json on summarize/stop.
- Coffee break added: pause the show, order coffee, resume. Saves ./sessions/<session_id>_coffee.json
"""
print("🔥 AGENT FILE LOADED:", __file__)

import json
import logging
import asyncio
import uuid
import random
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
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
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("voice_improv_battle")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Improv Scenarios (seeded)
# -------------------------
SCENARIOS = [
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are an overenthusiastic TV infomercial host selling a product that clearly does not work as advertised.",
    "You are an astronaut who just discovered the ship's coffee machine has developed a personality.",
    "You are a nervous wedding officiant who keeps getting the couple's names mixed up in ridiculous ways.",
    "You are a ghost trying to give a performance review to a living employee.",
    "You are a medieval king reacting to a very modern delivery service showing up at court.",
    "You are a detective interrogating a suspect who only answers in awkward metaphors."
]

# -------------------------
# Coffee Menu
# -------------------------
COFFEE_MENU = {
    "espresso":       {"name": "Espresso",            "price": 2.50, "emoji": "☕"},
    "latte":          {"name": "Latte",                "price": 3.50, "emoji": "🥛"},
    "cappuccino":     {"name": "Cappuccino",           "price": 3.50, "emoji": "☕"},
    "americano":      {"name": "Americano",            "price": 3.00, "emoji": "☕"},
    "cold brew":      {"name": "Cold Brew",            "price": 4.00, "emoji": "🧊"},
    "mocha":          {"name": "Mocha",                "price": 4.00, "emoji": "🍫"},
    "macchiato":      {"name": "Macchiato",            "price": 3.75, "emoji": "☕"},
    "hot chocolate":  {"name": "Hot Chocolate",        "price": 3.25, "emoji": "🍫"},
    "green tea":      {"name": "Green Tea",            "price": 2.75, "emoji": "🍵"},
    "chai latte":     {"name": "Chai Latte",           "price": 3.75, "emoji": "🍵"},
}

# -------------------------
# JSON Save Helpers
# -------------------------
SESSIONS_DIR = "sessions"

def _save_session_json(userdata: "Userdata") -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    filepath = os.path.join(SESSIONS_DIR, f"{userdata.session_id}.json")
    payload = {
        "session_id":   userdata.session_id,
        "player_name":  userdata.player_name,
        "started_at":   userdata.started_at,
        "saved_at":     datetime.utcnow().isoformat() + "Z",
        "improv_state": userdata.improv_state,
        "rounds": [
            {
                "round":       r.get("round"),
                "scenario":    r.get("scenario"),
                "performance": r.get("performance"),
                "reaction":    r.get("reaction"),
            }
            for r in userdata.improv_state.get("rounds", [])
        ],
        "history": userdata.history,
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Session saved → {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save session JSON: {e}")
        return f"(save failed: {e})"


def _save_coffee_json(userdata: "Userdata") -> str:
    """Save all coffee break orders for this session to a separate JSON file."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    filepath = os.path.join(SESSIONS_DIR, f"{userdata.session_id}_coffee.json")

    breaks = userdata.coffee_breaks
    total_spent = sum(b.get("total_price", 0) for b in breaks)
    total_drinks = sum(len(b.get("orders", [])) for b in breaks)

    payload = {
        "session_id":    userdata.session_id,
        "player_name":   userdata.player_name,
        "saved_at":      datetime.utcnow().isoformat() + "Z",
        "total_breaks":  len(breaks),
        "total_drinks":  total_drinks,
        "total_spent":   round(total_spent, 2),
        "currency":      "USD",
        "coffee_breaks": breaks,
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"☕ Coffee log saved → {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save coffee JSON: {e}")
        return f"(save failed: {e})"


# -------------------------
# Per-session Improv State
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    improv_state: Dict = field(default_factory=lambda: {
        "current_round": 0,
        "max_rounds": 3,
        "rounds": [],
        "phase": "idle",
        "used_indices": []
    })
    history: List[Dict] = field(default_factory=list)
    # ── Coffee break state ──────────────────────────────────────
    coffee_breaks: List[Dict] = field(default_factory=list)
    active_coffee_break: Optional[Dict] = None   # set while on break


# -------------------------
# Helpers
# -------------------------
def _pick_scenario(userdata: Userdata) -> str:
    used = userdata.improv_state.get("used_indices", [])
    candidates = [i for i in range(len(SCENARIOS)) if i not in used]
    if not candidates:
        userdata.improv_state["used_indices"] = []
        candidates = list(range(len(SCENARIOS)))
    idx = random.choice(candidates)
    userdata.improv_state["used_indices"].append(idx)
    return SCENARIOS[idx]


def _host_reaction_text(performance: str) -> str:
    tones = ["supportive", "neutral", "mildly_critical"]
    tone = random.choice(tones)
    highlights = []
    perf_l = (performance or "").lower()
    if any(w in perf_l for w in ("funny", "lol", "hahaha", "haha")):
        highlights.append("great comedic timing")
    if any(w in perf_l for w in ("sad", "cry", "tears")):
        highlights.append("good emotional depth")
    if "pause" in perf_l or "..." in perf_l:
        highlights.append("interesting use of silence")
    if not highlights:
        highlights.append(random.choice(["nice character choices", "bold commitment", "unexpected twist"]))
    chosen = random.choice(highlights)
    if tone == "supportive":
        return f"Neon MC: Love that — {chosen}! That was playful and clear. Nice work. Ready for the next beat?"
    elif tone == "neutral":
        return f"Neon MC: Hmm — {chosen}. Interesting shapes in there; try leaning more into one choice. Next scene when you're ready."
    else:
        return f"Neon MC: Okay — {chosen}, but that felt a bit rushed. Push the choices louder next time. Let's level up."


def _menu_text() -> str:
    lines = ["☕ Neon Arcade Coffee Bar — What'll it be?", ""]
    for key, item in COFFEE_MENU.items():
        lines.append(f"  {item['emoji']} {item['name']:20s} ${item['price']:.2f}")
    lines.append("\nJust say the drink name to order. Say 'done ordering' when finished.")
    return "\n".join(lines)


# -------------------------
# Agent Tools
# -------------------------
@function_tool
async def start_show(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Player/contestant name (optional)", default=None)] = None,
    max_rounds: Annotated[int, Field(description="Number of rounds (3-5 recommended)", default=3)] = 3,
) -> str:
    userdata = ctx.userdata
    if name:
        userdata.player_name = name.strip()
    else:
        userdata.player_name = userdata.player_name or "Contestant"

    max_rounds = max(1, min(int(max_rounds), 8))
    userdata.improv_state["max_rounds"] = max_rounds
    userdata.improv_state["current_round"] = 0
    userdata.improv_state["rounds"] = []
    userdata.improv_state["phase"] = "intro"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "start_show", "name": userdata.player_name})

    intro = (
        f"*** Welcome to Improv Battle — Neon Arcade Edition! ***\n"
        f"I'm Neon MC, your synth-powered host.\n"
        f"{userdata.player_name or 'Contestant'}, we're running {userdata.improv_state['max_rounds']} rounds.\n"
        "Rules: I'll flash a quick scene, you play it out. Say 'End scene' or pause when you're done — I'll react and move on. Keep it bold!\n"
        "💡 Tip: Say 'coffee break' anytime to pause and grab a drink!"
    )

    scenario = _pick_scenario(userdata)
    userdata.improv_state["current_round"] = 1
    userdata.improv_state["phase"] = "awaiting_improv"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "present_scenario", "round": 1, "scenario": scenario})

    return intro + "\n\nRound 1: " + scenario + "\n\nStart improvising now!"


@function_tool
async def coffee_break(
    ctx: RunContext[Userdata],
) -> str:
    """Pause the show and open the coffee bar so the player can order drinks."""
    userdata = ctx.userdata

    if userdata.active_coffee_break is not None:
        return "You're already on a coffee break! Order something or say 'done ordering' to resume."

    if userdata.improv_state.get("phase") == "done":
        return "The show is over, but the coffee bar is always open! (No active session to resume.)"

    # Freeze current phase
    userdata.improv_state["_phase_before_break"] = userdata.improv_state.get("phase", "idle")
    userdata.improv_state["phase"] = "coffee_break"

    # Open a new break record
    userdata.active_coffee_break = {
        "break_id":   str(uuid.uuid4())[:6],
        "started_at": datetime.utcnow().isoformat() + "Z",
        "ended_at":   None,
        "orders":     [],
        "total_price": 0.0,
    }
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "coffee_break_start"})

    return (
        "⏸  Show paused — enjoy your break!\n\n"
        + _menu_text()
    )


@function_tool
async def order_coffee(
    ctx: RunContext[Userdata],
    drink: Annotated[str, Field(description="Name of the drink to order (e.g. 'latte', 'espresso')")],
    quantity: Annotated[int, Field(description="How many of this drink", default=1)] = 1,
    customization: Annotated[Optional[str], Field(description="Optional customization e.g. 'oat milk, no sugar'", default=None)] = None,
) -> str:
    """Add a drink to the current coffee break order."""
    userdata = ctx.userdata

    if userdata.active_coffee_break is None:
        return "You're not on a coffee break right now. Say 'coffee break' to start one!"

    drink_key = drink.strip().lower()

    # Fuzzy match against menu keys
    matched_key = None
    for key in COFFEE_MENU:
        if drink_key in key or key in drink_key:
            matched_key = key
            break

    if not matched_key:
        menu_keys = ", ".join(COFFEE_MENU.keys())
        return f"Hmm, I don't recognise '{drink}'. We have: {menu_keys}. What would you like?"

    item = COFFEE_MENU[matched_key]
    quantity = max(1, min(int(quantity), 10))
    line_price = round(item["price"] * quantity, 2)

    order_entry = {
        "drink":          item["name"],
        "quantity":       quantity,
        "unit_price":     item["price"],
        "line_price":     line_price,
        "customization":  customization or "",
        "ordered_at":     datetime.utcnow().isoformat() + "Z",
    }

    userdata.active_coffee_break["orders"].append(order_entry)
    userdata.active_coffee_break["total_price"] = round(
        userdata.active_coffee_break["total_price"] + line_price, 2
    )

    custom_note = f" ({customization})" if customization else ""
    return (
        f"{item['emoji']} Got it — {quantity}x {item['name']}{custom_note} for ${line_price:.2f}.\n"
        f"Running total: ${userdata.active_coffee_break['total_price']:.2f}\n"
        "Anything else? Say another drink or 'done ordering' to resume the show."
    )


@function_tool
async def end_coffee_break(
    ctx: RunContext[Userdata],
) -> str:
    """Finish ordering and resume the improv show."""
    userdata = ctx.userdata

    if userdata.active_coffee_break is None:
        return "You're not on a coffee break. Say 'coffee break' to start one!"

    # Close the break
    break_record = userdata.active_coffee_break
    break_record["ended_at"] = datetime.utcnow().isoformat() + "Z"
    userdata.coffee_breaks.append(break_record)
    userdata.active_coffee_break = None

    # Restore previous phase
    prev_phase = userdata.improv_state.pop("_phase_before_break", "awaiting_improv")
    userdata.improv_state["phase"] = prev_phase

    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "coffee_break_end"})

    # Save coffee JSON immediately
    coffee_filepath = _save_coffee_json(userdata)

    orders = break_record.get("orders", [])
    if orders:
        order_summary = ", ".join(
            f"{o['quantity']}x {o['drink']}" + (f" ({o['customization']})" if o.get("customization") else "")
            for o in orders
        )
        receipt = (
            f"☕ Order summary: {order_summary}\n"
            f"💰 Total: ${break_record['total_price']:.2f}\n"
            f"📄 Coffee log saved → {coffee_filepath}\n"
        )
    else:
        receipt = "No drinks ordered this break. Staying hydrated the old-fashioned way!\n"

    # Figure out what to say next
    cur = userdata.improv_state.get("current_round", 0)
    phase = userdata.improv_state.get("phase")

    if phase == "awaiting_improv" and cur > 0:
        resume_msg = f"▶  Back to it! We're on Round {cur}. Pick up where you left off — go!"
    elif phase == "reacting":
        resume_msg = f"▶  Back to it! Say 'Next' whenever you're ready for Round {cur + 1}."
    else:
        resume_msg = "▶  Welcome back! Say 'Next' to continue the show."

    return receipt + resume_msg


@function_tool
async def next_scenario(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    if userdata.improv_state.get("phase") == "coffee_break":
        return "You're still on a coffee break! Say 'done ordering' to resume first."
    if userdata.improv_state.get("phase") == "done":
        return "The show is already over. Say 'start show' to play again."

    cur = userdata.improv_state.get("current_round", 0)
    maxr = userdata.improv_state.get("max_rounds", 3)
    if cur >= maxr:
        userdata.improv_state["phase"] = "done"
        return await summarize_show(ctx)

    next_round = cur + 1
    scenario = _pick_scenario(userdata)
    userdata.improv_state["current_round"] = next_round
    userdata.improv_state["phase"] = "awaiting_improv"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "present_scenario", "round": next_round, "scenario": scenario})
    return f"Round {next_round}: {scenario}\nGo!"


@function_tool
async def record_performance(
    ctx: RunContext[Userdata],
    performance: Annotated[str, Field(description="Player's improv performance (transcribed text)")],
) -> str:
    userdata = ctx.userdata

    if userdata.improv_state.get("phase") == "coffee_break":
        return "You're on a coffee break! Say 'done ordering' to resume before performing."

    if userdata.improv_state.get("phase") != "awaiting_improv":
        userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "record_performance_out_of_phase"})

    round_no = userdata.improv_state.get("current_round", 0)
    scenario = userdata.history[-1].get("scenario") if userdata.history and userdata.history[-1].get("action") == "present_scenario" else "(unknown)"
    reaction = _host_reaction_text(performance)

    userdata.improv_state["rounds"].append({
        "round": round_no,
        "scenario": scenario,
        "performance": performance,
        "reaction": reaction,
    })
    userdata.improv_state["phase"] = "reacting"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "record_performance", "round": round_no})

    if round_no >= userdata.improv_state.get("max_rounds", 3):
        userdata.improv_state["phase"] = "done"
        closing = "\n" + reaction + "\nThat's the final round. "
        closing += (await summarize_show(ctx))
        return closing

    closing = reaction + "\nWhen you're ready, say 'Next' or I'll spin up the next scene."
    return closing


@function_tool
async def summarize_show(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    rounds = userdata.improv_state.get("rounds", [])
    if not rounds:
        return "No rounds were played. Thanks for dropping into Improv Battle!"

    summary_lines = [f"Thanks for playing, {userdata.player_name or 'Contestant'}! Here's your Neon Arcade recap:"]
    for r in rounds:
        perf_snip = (r.get("performance") or "").strip()
        if len(perf_snip) > 80:
            perf_snip = perf_snip[:77] + "..."
        summary_lines.append(f"Round {r.get('round')}: {r.get('scenario')} — You: '{perf_snip}' | Host: {r.get('reaction')}")

    mentions_character = sum(1 for r in rounds if any(w in (r.get('performance') or '').lower() for w in ('i am', "i'm", 'as a', 'character', 'role')))
    mentions_emotion = sum(1 for r in rounds if any(w in (r.get('performance') or '').lower() for w in ('sad', 'angry', 'happy', 'love', 'cry', 'tears')))

    profile = "You seem like a player who "
    if mentions_character > len(rounds) / 2:
        profile += "commits to character choices"
    elif mentions_emotion > 0:
        profile += "brings emotional color to scenes"
    else:
        profile += "likes surprising beats and twists"
    profile += ". Keep pushing the choices and have fun."

    summary_lines.append(profile)

    # Coffee summary if any breaks taken
    if userdata.coffee_breaks:
        total_drinks = sum(len(b.get("orders", [])) for b in userdata.coffee_breaks)
        total_spent  = sum(b.get("total_price", 0) for b in userdata.coffee_breaks)
        summary_lines.append(
            f"☕ Coffee corner: {len(userdata.coffee_breaks)} break(s), "
            f"{total_drinks} drink(s), ${total_spent:.2f} total."
        )

    summary_lines.append("Neon MC: Thanks for performing on Improv Battle — keep the synth alive!")

    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "summarize_show"})

    # Save both JSONs
    filepath = _save_session_json(userdata)
    summary_lines.append(f"(Session saved → {filepath})")

    if userdata.coffee_breaks:
        coffee_filepath = _save_coffee_json(userdata)
        summary_lines.append(f"(Coffee log saved → {coffee_filepath})")

    return "\n".join(summary_lines)


@function_tool
async def stop_show(ctx: RunContext[Userdata], confirm: Annotated[bool, Field(description="Confirm stop", default=False)] = False) -> str:
    userdata = ctx.userdata
    if not confirm:
        return "Are you sure you want to stop the show? Say 'stop show yes' to confirm."
    userdata.improv_state["phase"] = "done"
    userdata.history.append({"time": datetime.utcnow().isoformat() + "Z", "action": "stop_show"})

    filepath = _save_session_json(userdata)
    msg = f"Show stopped. Thanks for visiting Neon Arcade Improv Battle! (Session saved → {filepath})"

    if userdata.coffee_breaks:
        coffee_filepath = _save_coffee_json(userdata)
        msg += f"\n(Coffee log saved → {coffee_filepath})"

    return msg


# -------------------------
# The Agent (Improv Host)
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        instructions = """
        You are the host of a TV improv show called 'Improv Battle' — Neon Arcade Edition.
        Role: High-energy, witty synthwave MC. Guide a single contestant through short improv scenes.

        Behavioural rules:
            - Introduce the show and explain the rules at the start (use the retro/arcade flavor).
            - Present clear scenario prompts (who you are, what's happening, what's the tension).
            - Prompt the player to improvise and listen for "End scene" or accept an utterance passed to record_performance.
            - After each scene, react in a varied, realistic way (supportive, neutral, mildly critical). Store the reaction.
            - Run configured number of rounds, then summarize the player's style.
            - Keep turns short and TTS-friendly.
            - If the player says "coffee break", "I need a break", or "let's get coffee" → call coffee_break tool.
            - If the player orders a drink by name → call order_coffee tool.
            - If the player says "done ordering", "that's all", "resume" or "back to the show" → call end_coffee_break tool.
        Use tools: start_show, next_scenario, record_performance, summarize_show, stop_show,
                   coffee_break, order_coffee, end_coffee_break.
        """
        super().__init__(
            instructions=instructions,
            tools=[
                start_show, next_scenario, record_performance,
                summarize_show, stop_show,
                coffee_break, order_coffee, end_coffee_break,   # ← NEW
            ],
        )


# -------------------------
# Entrypoint & Prewarm
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🎮" * 6)
    logger.info("🚀 STARTING NEON ARCADE VOICE HOST — Improv Battle")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
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
