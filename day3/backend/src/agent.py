import logging
import json
import time
import uuid
import os
from datetime import datetime, timedelta
from threading import Lock
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# File configuration
LOG_FILE = "wellness_log.json"
_file_lock = Lock()

def get_last_session_context():
    """Reads the last line of the JSON file to provide context for the prompt."""
    if not os.path.exists(LOG_FILE):
        return "This is the user's first session. Welcome them warmly."
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return "First session."
            
            # Parse the last line
            last_entry = json.loads(lines[-1])
            date = last_entry.get("date", "unknown")
            goals = last_entry.get("goals", [])
            mood_score = last_entry.get("mood_score", "unknown")
            mood_text = last_entry.get("mood_text", "unknown")
            
            return f"Last session was on {date}. Mood: {mood_text} ({mood_score}/10). Goals set: {', '.join(goals)}."
    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return "Error reading history. Treat as fresh session."

class WellnessCompanion(Agent):
    def __init__(self, past_context: str) -> None:
        super().__init__(
            instructions=f"""
You are Sam, a secure and private AI wellness companion. 
Your ID is {str(uuid.uuid4())[:4]}.

**CRITICAL SPEAKING STYLE (MUST FOLLOW):**
- **No Symbols:** Never use slashes (/) or brackets ([]). 
- **Natural Numbers:** Instead of "8/10", write "8 out of 10". Instead of "1-3 goals", write "one to three goals".
- **Warm Tone:** Speak naturally, like a podcast host, not a robot reading a report.

**CRITICAL SAFETY PROTOCOL:**
If the user mentions self-harm, suicide, or severe physical pain:
1. STOP immediately.
2. Say: "I am an AI, not a doctor. Please call emergency services or a crisis hotline immediately."
3. Do NOT offer further wellness advice.

**YOUR CONVERSATION FLOW:**
1. **Disclaimer:** Start by briefly stating: "I'm Sam, your AI companion."
2. **Check-in:** Ask "How are you feeling today on a scale of 1 to 10?" If the score is low (5 or below), gently ask: "I'm sorry to hear that. Is there anything specific on your mind, or just a low energy day?" Listen to their answer before asking for goals.
3. **Reflect:** If the user asks "How was my week?", call `analyze_my_week`.
4. **Goals:** Ask for 1 to 3 simple goals for today.
5. **Support:** Offer ONE simple, grounded tip.
6. **Confirm:** Recap mood and goals. (Say "You are feeling an 8 out of 10", NOT "8/10").
7. **Save:** Once they confirm, call `save_checkin`.
**IMPORTANT - AFTER SAVING:**
- Once `save_checkin` is called, the daily check-in is COMPLETE.
- If the user keeps talking (e.g., "How was my week?"), answer their question ONLY.
- **DO NOT** ask "How are you feeling?" or "What are your goals?" again.
- Just answer and politely ask: "Is there anything else I can help with?"

**CONTEXT FROM LAST TIME:**
{past_context}
"""
        )

        # Current session state
        self.wellness_state = {
            "mood_text": None,
            "mood_score": None, # Numeric 1-10 for aggregation
            "goals": [],
            "summary": None
        }

    @function_tool
    async def update_mood(self, ctx: RunContext, mood_text: str, score: int):
        """
        Records the mood.
        score: An integer from 1 (Terrible) to 10 (Amazing).
        mood_text: One or two words describing the feeling (e.g., "Tired", "Hopeful").
        """
        self.wellness_state["mood_text"] = mood_text
        self.wellness_state["mood_score"] = score
        return {"ok": True, "msg": f"Recorded mood: {mood_text} ({score}/10)"}

    @function_tool
    async def add_goal(self, ctx: RunContext, goal: str):
        """Adds a goal to the list. Call multiple times or comma separate for multiple goals."""
        if "," in goal:
            goals = [g.strip() for g in goal.split(",")]
            self.wellness_state["goals"].extend(goals)
        else:
            self.wellness_state["goals"].append(goal)
        return {"ok": True, "goals": self.wellness_state["goals"]}

    @function_tool
    async def analyze_my_week(self, ctx: RunContext):
        """
        ADVANCED TOOL: Reads the JSON history to calculate weekly stats.
        Use this when the user asks "How has my week been?" or "How am I doing?".
        """
        if not os.path.exists(LOG_FILE):
            return "No history available yet."

        try:
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            
            total_score = 0
            count = 0
            days_with_goals = 0
            
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_date = datetime.strptime(entry["date"], "%Y-%m-%d %H:%M:%S")
                        
                        # Filter for last 7 days
                        if entry_date >= week_ago:
                            score = entry.get("mood_score")
                            if score is not None:
                                total_score += int(score)
                                count += 1
                            
                            if entry.get("goals") and len(entry["goals"]) > 0:
                                days_with_goals += 1
                    except Exception:
                        continue # Skip bad lines

            if count == 0:
                return "No entries found for the last 7 days."

            avg_mood = round(total_score / count, 1)
            
            return f"""
            WEEKLY REPORT:
            - Entries: {count}
            - Average Mood: {avg_mood}/10
            - Goal Consistency: You set goals on {days_with_goals} days this week.
            """
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return "Could not generate report due to an error."

    @function_tool
    async def save_checkin(self, ctx: RunContext, summary_note: str):
        """Saves session and ends interaction. summary_note is a 1-sentence recap."""
        self.wellness_state["summary"] = summary_note
        self.wellness_state["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.wellness_state["id"] = str(uuid.uuid4())

        if not self.wellness_state["mood_score"]:
            self.wellness_state["mood_score"] = 5 # Default neutral if unspecified

        try:
            with _file_lock:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self.wellness_state, ensure_ascii=False) + "\n")
            return {"ok": True, "msg": "Saved. Goodbye."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # 1. LOAD CONTEXT (The Memory)
    past_context = get_last_session_context()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-001"), 
        tts=murf.TTS(
            model="falcon",
            voice="en-US-Ken", 
            style="Conversational",
            # REMOVE or COMMENT OUT the tokenizer line below to use the default streaming behavior
            # tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2), 
            text_pacing=False
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
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

    # 2. START AGENT
    await session.start(
        agent=WellnessCompanion(past_context=past_context),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))