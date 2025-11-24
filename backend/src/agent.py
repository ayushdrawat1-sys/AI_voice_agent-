import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Consolidated typing imports (fallback for Annotated)
try:
    from typing import Annotated, Literal, List, Optional
except Exception:
    from typing_extensions import Annotated  # type: ignore
    from typing import Literal, List, Optional

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
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("wellness_agent")
load_dotenv(".env.local")

# ---------------------------
# Persistence / file handling
# ---------------------------
WELLNESS_LOG_FILENAME = "wellness_log.json"


def get_log_path() -> str:
    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    return os.path.join(backend_dir, WELLNESS_LOG_FILENAME)


def load_history() -> List[dict]:
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        logger.warning("Could not read history file: %s", e)
        return []


def load_persona():
    # try multiple likely locations & filenames
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "agent_persona.json"),
        os.path.join(os.path.dirname(__file__), "..", "agent_maya.json"),
        os.path.join(os.path.dirname(__file__), "agent_persona.json"),
        os.path.join(os.path.dirname(__file__), "agent_maya.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    persona = json.load(f)
                    logger.info("Loaded persona from %s", p)
                    return persona
            except Exception as e:
                logger.error("Could not read persona file %s: %s", p, e)
    logger.info("No persona file found, using defaults.")
    return None


def append_checkin_record(record: dict) -> None:
    path = get_log_path()
    history = load_history()
    history.append(record)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    logger.info("Saved check-in to %s", path)


# ---------------------------
# Data classes for runtime
# ---------------------------
@dataclass
class CheckInState:
    mood: str | None = None
    energy: str | None = None
    objectives: List[str] = field(default_factory=list)
    advice_given: str | None = None

    def is_complete(self) -> bool:
        return bool(self.mood and self.energy and len(self.objectives) >= 1)

    def to_record(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "mood": self.mood,
            "energy": self.energy,
            "objectives": self.objectives,
            "summary": self.advice_given,
        }


@dataclass
class Userdata:
    current_checkin: CheckInState
    history_summary: str


# ---------------------------
# Tools (used by the agent)
# ---------------------------

@function_tool
async def record_mood_and_energy(
    ctx: RunContext[Userdata],
    mood: Annotated[str, Field(description="User mood")],
    energy: Annotated[str, Field(description="Energy level")],
) -> str:
    ctx.userdata.current_checkin.mood = mood.strip()
    ctx.userdata.current_checkin.energy = energy.strip()
    logger.debug("Mood logged: %s | Energy: %s", mood, energy)
    return f"Thanks — I noted you're feeling {mood} with {energy} energy."


@function_tool
async def record_objectives(
    ctx: RunContext[Userdata],
    objectives: Annotated[List[str], Field(description="1-3 objectives for today")],
) -> str:
    clean = [o.strip() for o in objectives if o and o.strip()][:3]
    ctx.userdata.current_checkin.objectives = clean
    logger.debug("Objectives logged: %s", clean)
    return "Got it — I've saved your objectives for today."


@function_tool
async def complete_checkin(
    ctx: RunContext[Userdata],
    final_advice_summary: Annotated[str, Field(description="Short (1-sentence) agent summary/advice")],
) -> str:
    state = ctx.userdata.current_checkin
    state.advice_given = final_advice_summary.strip()
    if not state.is_complete():
        return "I can't finish yet — I still need your mood, energy, or at least one objective."

    record = state.to_record()
    append_checkin_record(record)

    recap = (
        f"Recap: You're feeling {state.mood} with {state.energy} energy. "
        f"Today's objectives: {', '.join(state.objectives)}. "
        f"My note: {state.advice_given} "
        "Does this sound right?"
    )
    logger.info("Check-in completed: %s", record)
    return recap


# ---------------------------
# Agent definition
# ---------------------------


def build_system_instructions(history_summary: str, personality: str | None = None, persona_obj: dict | None = None) -> str:
    persona_text = f"\nPersonality: {personality}\n" if personality else "\nPersonality: calm, kind, concise.\n"

    name_text = ""
    greeting_text = ""
    if persona_obj:
        if persona_obj.get("name"):
            name_text = f"\nAgent name: {persona_obj['name']}\n"
        if persona_obj.get("greeting"):
            greeting_text = f"\nSuggested greeting: {persona_obj['greeting']}\n"

    instructions = f"""
You are a compassionate, supportive Daily Wellness Companion. Keep interactions short and grounded.

Context from previous sessions:
{history_summary}

Session goals:
1) Ask about mood and energy (example: "How are you feeling today? What's your energy like?")
2) Ask for 1–3 simple objectives for today.
3) Offer short, non-medical, actionable suggestions (e.g., 5-minute walk, break task into small steps).
4) Provide a brief recap and ask "Does this sound right?" then call complete_checkin with a 1-sentence summary.

Safety:
- Do NOT diagnose or give medical advice.
- If user expresses self-harm or severe crisis, suggest contacting a professional or emergency help.

{persona_text}
{name_text}
{greeting_text}

Use the agent tools to store mood, energy, objectives, and finalize the check-in.
"""
    return instructions


class WellnessAgent(Agent):
    def __init__(self, history_context: str, personality: str | None = None, persona_obj: dict | None = None):
        super().__init__(
            instructions=build_system_instructions(history_context, personality, persona_obj),
            tools=[record_mood_and_energy, record_objectives, complete_checkin],
        )


# ---------------------------
# Entrypoint & session startup
# ---------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    persona = load_persona()
    logger.info("Starting wellness session in room %s", ctx.room.name)
    logger.info("Persona loaded: %s", persona)

    history = load_history()
    if history:
        last = history[-1]
        history_summary = (
            f"Last check-in on {last.get('timestamp','unknown')}: mood={last.get('mood')}, energy={last.get('energy')}. "
            f"Objectives: {', '.join(last.get('objectives', []))}."
        )
    else:
        history_summary = "No previous history found."

    userdata = Userdata(current_checkin=CheckInState(), history_summary=history_summary)

    # livekit session config (adjust models/voice as you prefer)
    tts_settings = persona.get("tts") if persona else {}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=tts_settings.get("voice", "en-US-natalie"),
            style=tts_settings.get("style", "Soft"),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    # Build agent with history context (optionally pass personality string + full persona)
    agent = WellnessAgent(
        history_context=history_summary,
        personality=persona.get("tone") if persona else None,
        persona_obj=persona,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
