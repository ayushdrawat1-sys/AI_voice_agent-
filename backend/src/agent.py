#!/usr/bin/env python3
"""
Day 4 - Teach-the-Tutor agent (merged & cleaned)

Features:
- Small content JSON (shared-data/day4_tutor_content.json)
- Modes: learn (Matthew), quiz (Alicia), teach_back (Ken)
- Persona loader (Maya) introduces the three coaches
- Tools: select_topic, set_learning_mode, evaluate_teaching
- Simple keyword-based evaluation for teach_back
"""

import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Literal

# Typing fallback/compat (Annotated is only needed for pydantic fields)
try:
    from typing import Annotated
except Exception:
    from typing_extensions import Annotated  # type: ignore

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
    RunContext,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("day4_tutor")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")

# -----------------------------
# Content file (small JSON)
# -----------------------------
CONTENT_PATH = os.path.join("shared-data", "day4_tutor_content.json")
DEFAULT_CONTENT = [
    {
        "id": "variables",
        "title": "Variables",
        "summary": "Variables store values so you can reuse them later. They have a name and a value; values can change during execution.",
        "sample_question": "What is a variable and why is it useful?"
    },
    {
        "id": "loops",
        "title": "Loops",
        "summary": "Loops let you repeat an action multiple times. Common varieties are for-loops (repeat a set number of times) and while-loops (repeat while a condition is true).",
        "sample_question": "Explain the difference between a for loop and a while loop."
    }
]


def ensure_content_file(path: str = CONTENT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONTENT, f, indent=2, ensure_ascii=False)
        logger.info("Created default content file at %s", path)


def load_content(path: str = CONTENT_PATH) -> List[dict]:
    try:
        ensure_content_file(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("Content file not a list, using defaults.")
            return DEFAULT_CONTENT
    except Exception as e:
        logger.exception("Failed to load content file: %s", e)
        return DEFAULT_CONTENT


COURSE_CONTENT = load_content()

# -----------------------------
# Persona & persistence utils
# -----------------------------
def load_persona():
    # Try several likely persona filenames (user supplied)
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

# -----------------------------
# Runtime state dataclasses
# -----------------------------
@dataclass
class TutorState:
    current_topic_id: Optional[str] = None
    current_topic_data: Optional[dict] = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str, content_list: List[dict]) -> bool:
        tid = topic_id.lower()
        topic = next((t for t in content_list if t["id"].lower() == tid), None)
        if topic:
            self.current_topic_id = topic["id"]
            self.current_topic_data = topic
            return True
        return False


@dataclass
class Userdata:
    tutor_state: TutorState = field(default_factory=TutorState)
    agent_session: Optional[AgentSession] = None
    # optional: track mastery (concept_id -> score)
    mastery: dict = field(default_factory=dict)


# -----------------------------
# Tools: select_topic, set_learning_mode, evaluate_teaching
# -----------------------------
@function_tool
async def select_topic(
    ctx: RunContext[Userdata],
    topic_id: Annotated[str, Field(description="The ID of the topic to study (e.g., 'variables', 'loops')")]
) -> str:
    """Select a topic by id (uses COURSE_CONTENT)."""
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id, COURSE_CONTENT)
    if success:
        title = state.current_topic_data["title"]
        return f"Topic set to '{title}'. Ask to 'Learn', 'Quiz' or 'Teach it back' when you're ready."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic '{topic_id}' not found. Available topics: {available}."


@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata],
    mode: Annotated[str, Field(description="Mode to switch to: 'learn', 'quiz', 'teach_back'")]
) -> str:
    """Switches learning mode and updates voice/persona for the agent session."""
    m = mode.lower()
    if m not in ("learn", "quiz", "teach_back"):
        return "Invalid mode. Choose one of: learn, quiz, teach_back."

    state = ctx.userdata.tutor_state
    state.mode = m

    # Update TTS voice to match mode (if session exists)
    session = ctx.userdata.agent_session
    if session:
        # Default Murf voices (these labels may vary by account; adjust as needed)
        if m == "learn":
            # Matthew: the explainer
            _voice = "en-US-matthew"
            _style = "Promo"
            instruction = f"Mode: LEARN. Explain: {state.current_topic_data.get('summary', '[no topic selected]')}"
        elif m == "quiz":
            # Alicia: the examiner
            _voice = "en-US-alicia"
            _style = "Conversational"
            instruction = f"Mode: QUIZ. Ask: {state.current_topic_data.get('sample_question', '[no topic selected]')}"
        else:  # teach_back
            # Ken: the learner/student
            _voice = "en-US-ken"
            _style = "Promo"
            instruction = "Mode: TEACH_BACK. Prompt the user to explain the concept back to you."

        try:
            # update_options may vary by TTS implementation; using same pattern as your sample
            session.tts.update_options(voice=_voice, style=_style)
            logger.info("Switched TTS to %s / %s for mode %s", _voice, _style, m)
        except Exception as e:
            logger.warning("Failed to update tts options: %s", e)
    else:
        instruction = "Mode switched but agent session not available for voice change."

    return f"Switched to mode '{m}'. {instruction}"


def _simple_score_and_feedback(topic_summary: str, user_text: str) -> (int, str):
    """
    Very simple evaluator:
    - counts overlap of important words (split summary into tokens, exclude common words)
    - returns score 0-10 and short feedback text
    This is intentionally basic — replace with LLM evaluation if you want richer feedback.
    """
    if not topic_summary:
        return 0, "No topic summary available to evaluate against."

    def tokenize(s):
        return [t.strip().lower() for t in s.replace("(", " ").replace(")", " ").replace(",", " ").split() if len(t) > 2]

    summary_tokens = set(tokenize(topic_summary))
    user_tokens = set(tokenize(user_text))
    if not user_tokens:
        return 0, "I didn't catch any explanation — try explaining the concept in your own words."

    matches = summary_tokens.intersection(user_tokens)
    # score proportional to fraction of summary tokens covered
    if summary_tokens:
        frac = len(matches) / max(1, len(summary_tokens))
    else:
        frac = 0.0
    score = int(round(frac * 10))
    # qualitative feedback
    if score >= 8:
        fb = "Excellent — you covered the core ideas clearly. A few extra examples would make it perfect."
    elif score >= 5:
        fb = "Good — you captured several key points but missed some details."
    elif score >= 2:
        fb = "Partial — you mentioned a couple ideas but left out important parts. Try focusing on the main purpose and examples."
    else:
        fb = "Needs work — try to restate the main idea and one example or use-case."

    return max(0, min(10, score)), fb


@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="User's explanation during teach_back mode")]
) -> str:
    """Evaluate the user's teach-back explanation and update a simple mastery tracker."""
    state = ctx.userdata.tutor_state
    topic = state.current_topic_data
    if not topic:
        return "No topic selected — select a topic first with select_topic."

    summary = topic.get("summary", "")
    score, feedback = _simple_score_and_feedback(summary, user_explanation)

    # Update simple mastery (store best score)
    mastery = ctx.userdata.mastery
    tid = topic["id"]
    prev = mastery.get(tid, 0)
    mastery[tid] = max(prev, score)

    result = (
        f"Score: {score}/10.\n"
        f"Feedback: {feedback}\n"
        f"Topic mastery (best so far): {mastery[tid]}/10."
    )

    # Optionally, append a short record to disk (lightweight)
    try:
        rec = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "topic": tid,
            "score": score,
            "feedback": feedback,
        }
        logdir = os.path.join("shared-data", "tutor_logs")
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "teach_back_history.json"), "a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write teach_back history line.")

    return result


# -----------------------------
# Agent implementation
# -----------------------------
class TutorAgent(Agent):
    def __init__(self, persona_obj: Optional[dict] = None):
        topic_list = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])
        persona_text = ""
        if persona_obj:
            persona_text = f"\nPersona loaded: name={persona_obj.get('name')}, tone={persona_obj.get('tone')}\n"

        instructions = f"""
You are a Teach-The-Tutor coach. Available topics: {topic_list}

Modes:
- LEARN (voice: Matthew): explain the concept using the provided summary.
- QUIZ  (voice: Alicia): ask the concept's sample_question to the user.
- TEACH_BACK (voice: Ken): ask the user to explain the concept back, then evaluate.

Behavior:
- Start by greeting the user and asking which topic they'd like to study (or offer the available topics).
- Use the provided tools (select_topic, set_learning_mode, evaluate_teaching) to manage state and mode switching.
- Allow the user to switch modes anytime by asking to switch.
{persona_text}
"""

        super().__init__(instructions=instructions, tools=[select_topic, set_learning_mode, evaluate_teaching])


# -----------------------------
# Entrypoint & session startup
# -----------------------------
def prewarm(proc: JobProcess):
    # preload VAD model
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarmed VAD")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Starting Day4 tutor session in room %s", ctx.room.name)

    persona = load_persona() or {}
    # choose initial TTS voice from persona if available, else a Maya fallback (may need to match your Murf set)
    tts_settings = persona.get("tts", {})
    maya_voice = tts_settings.get("voice", "en-US-maya")  # default fallback label

    # Initialize userdata & session
    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=maya_voice,
            style=tts_settings.get("style", "Friendly"),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    # attach session to userdata so tools can update tts
    userdata.agent_session = session

    # Create agent, passing persona so the system prompt can reference Maya
    agent = TutorAgent(persona_obj=persona)

    # Start the interactive session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # After connection, have Maya greet and introduce the three coaches (Matthew, Alicia, Ken)
    maya_intro = persona.get("greeting") if persona and persona.get("greeting") else (
        "Hi — I'm Maya, your coach persona for today. "
        "I will introduce three specialized voices who will help you learn:\n"
        "- Matthew (the explainer) will be your teacher in Learn mode,\n"
        "- Alicia (the examiner) will quiz you in Quiz mode,\n"
        "- Ken (the student) will roleplay as a beginner during Teach-back mode.\n"
        "Which topic would you like to study? You can say a topic id (like 'variables' or 'loops') "
        "or ask me to list available topics."
    )

    try:
        # speak the intro using the session TTS (session.start should have begun streaming; this instruction
        # pattern depends on livekit client behavior; many implementations accept an initial system message
        # or the LLM will prompt the user. We'll ask the agent's LLM to say the greeting by sending a message.
        await session.say(maya_intro)
        logger.info("Maya intro spoken.")
    except Exception:
        logger.exception("Failed to send Maya greeting via session.say (this call may vary by SDK).")

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
