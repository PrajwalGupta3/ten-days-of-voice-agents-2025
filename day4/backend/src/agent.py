import logging
import json
import os
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    function_tool,
    RunContext,
    MetricsCollectedEvent,
    RoomInputOptions
)
from livekit.plugins import murf, deepgram, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("active-recall-coach")
load_dotenv(".env.local")

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
CONTENT_FILE = os.path.join(current_dir, "day4_tutor_content.json")

# Voice Mapping Configuration
VOICE_MAP = {
    "learn": "en-US-Matthew",  # Deep, explanatory voice
    "quiz": "en-US-Alicia",    # Sharp, questioning voice
    "teach_back": "en-US-Ken"  # Supportive, listener voice
}

def load_content():
    """Loads the course content from JSON."""
    try:
        if not os.path.exists(CONTENT_FILE):
            logger.error(f"CRITICAL: Content file not found at: {CONTENT_FILE}")
            return []
            
        with open(CONTENT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load content: {e}")
        return []

class ActiveRecallCoach(Agent):
    def __init__(self, session: AgentSession, content: list) -> None:
        self._session = session
        self._content = content
        self._current_mode = "setup"

        # Create a string representation of the topics for the prompt
        if content:
            topics_str = "\n".join([f"- {c['title']} (ID: {c['id']})" for c in content])
        else:
            topics_str = "No topics available. Please check the content file."

        super().__init__(
            instructions=f"""
You are a sophisticated Active Recall Coach. Your goal is to help the user master concepts by switching between teaching, quizzing, and listening.

**AVAILABLE TOPICS:**
{topics_str}

**YOUR MODES:**
1. **LEARN Mode (Voice: Matthew):** You explain the concept clearly using the 'summary' data.
2. **QUIZ Mode (Voice: Alicia):** You ask the user specific questions using 'sample_question' or by generating new ones based on the topic.
3. **TEACH_BACK Mode (Voice: Ken):** You ask the user to explain the concept to YOU. You listen, then give a score (1-5) and feedback on their explanation.

**CURRENT STATE:**
- When the conversation starts, greet the user warmly and ask which MODE they want to start with (Learn, Quiz, or Teach-Back) and which TOPIC.
- **IMPORTANT:** When the user selects a mode, you MUST call the `set_mode` tool immediately.
- If the tool says "Voice switch failed", **IGNORE the error** and proceed with the mode (Quiz/Learn/Teach) using your current voice. Do not stop.

**BEHAVIOR:**
- Keep responses concise.
- In Teach-Back mode, be encouraging but point out if they missed key details from the 'summary'.
"""
        )

    @function_tool
    async def set_mode(self, ctx: RunContext, mode: str):
        """
        Switches the learning mode and attempts to update the voice.
        mode: Must be one of 'learn', 'quiz', 'teach_back'.
        """
        mode = mode.lower()
        if mode not in VOICE_MAP:
            return f"Invalid mode. Please choose learn, quiz, or teach_back."
        
        self._current_mode = mode
        new_voice_id = VOICE_MAP[mode]
        status_msg = f"Mode logic switched to {mode.upper()}. "

        try:
            logger.info(f"Attempting to switch voice to {new_voice_id}")

            # 1. Create new TTS instance
            new_tts = murf.TTS(
                model="falcon",
                voice=new_voice_id,
                style="Conversational",
                text_pacing=False
            )
            
            # 2. FORCE update the private attribute (The "Power Move")
            # This bypasses the read-only property restriction of AgentSession
            self._session._tts = new_tts
            
            status_msg += f"Voice successfully updated to {new_voice_id}."

        except Exception as e:
            # 3. Graceful Fallback: Log error but tell LLM to keep going
            logger.error(f"Voice switch failed: {e}")
            status_msg += "Voice switch encountered an error, but I am ready to proceed with the content in this mode."

        return status_msg

    @function_tool
    async def get_concept_details(self, ctx: RunContext, topic_id: str):
        """Retrieves details for a specific topic ID (e.g., 'variables', 'loops')."""
        for item in self._content:
            if item["id"] == topic_id or item["title"].lower() == topic_id.lower():
                return json.dumps(item)
        return "Topic not found."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # 1. Load Content
    course_content = load_content()

    # 2. Initialize TTS
    initial_tts = murf.TTS(
        model="falcon",
        voice="en-US-Ken",
        style="Conversational",
        text_pacing=False
    )

    # 3. Initialize Session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-001"),
        tts=initial_tts,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # 4. Start the Active Recall Coach
    await session.start(
        agent=ActiveRecallCoach(session=session, content=course_content),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))