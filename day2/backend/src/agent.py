import logging
import json
import time
import uuid
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


# File & lock for safe writes
ORDERS_FILE = "orders.json"
_orders_file_lock = Lock()

# Canonical options (menu)
VALID_SIZES = {
    "tall": "Tall",
    "small": "Tall",
    "grande": "Grande",
    "medium": "Grande",
    "venti": "Venti",
    "large": "Venti",
    "trenta": "Trenta",
    "extra large": "Trenta",
}
VALID_MILKS = {
    "2%": "2%",
    "2 percent": "2%",
    "whole": "Whole",
    "non-fat": "Non-fat",
    "nonfat": "Non-fat",
    "oat": "Oat",
    "almond": "Almond",
    "soy": "Soy",
    "coconut": "Coconut",
}
VALID_EXTRAS = {
    "vanilla": "Vanilla",
    "caramel": "Caramel",
    "hazelnut": "Hazelnut",
    "cold foam": "Cold Foam",
    "extra shot": "Extra Shot",
    "shot": "Extra Shot",
    "whipped cream": "Whipped Cream",
    "whip": "Whipped Cream",
}

# Items that clearly are not a drink — quick blacklist
INVALID_DRINK_TERMS = {"beer", "pizza", "wine", "burger", "fries"}


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are Sam, a warm, high-energy barista at a coffee shop.
Use the provided tools to collect an order using the schema:
{
  "drinkType": "string",
  "size": "string",
  "milk": "string",
  "extras": ["string"],
  "name": "string"
}

RULES (must follow):
-Begin a conversation with "Welcome to starbucks.what can i get you?"
- Ask exactly one question at a time until all required fields are filled.
- Always use update_order(field, value) to update fields; do NOT claim a field was updated unless you called the tool.
- Use get_missing_fields() to determine which fields remain.
- When all required fields are present, call finalize_order() to save the order.
- If user provides multiple pieces of info in one utterance, update the relevant fields via update_order() for each piece found.
- If user asks "what milks do you have", list the menu options.
- If user requests an invalid item (e.g. "pizza", "beer"), politely refuse: "We only serve coffee and pastries here!"
- If user asks for a size not in menu, respond: "I wish we had that! But the biggest I can do is a Trenta. Should we go with that?"
- Keep replies short, friendly, and clear.
"""
        )

        # per-instance order state (prevent cross-session leakage)
        self.order_state = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None,
        }

    # -------------------------
    # Helper: compute missing fields
    # -------------------------
    def _missing_fields(self):
        # extras is optional (tweak here if you want extras required)
        missing = []
        if not self.order_state.get("drinkType"):
            missing.append("drinkType")
        if not self.order_state.get("size"):
            missing.append("size")
        if not self.order_state.get("milk"):
            missing.append("milk")
        # extras optional: do not include if empty
        if not self.order_state.get("name"):
            missing.append("name")
        return missing

    # -------------------------
    # TOOL: update_order
    # -------------------------
    @function_tool
    async def update_order(self, ctx: RunContext, field: str, value: str):
        """
        Updates a field in the order. Returns structured result:
        {"ok": True, "order_state": {...}} or {"ok": False, "error": "..."}
        Normalizes sizes, milks and extras.
        """
        try:
            field = field.strip()
            if field not in ("drinkType", "size", "milk", "extras", "name"):
                return {"ok": False, "error": f"Unknown field: {field}"}

            if field == "size":
                if not isinstance(value, str) or not value.strip():
                    return {"ok": False, "error": "Empty size value"}
                v = value.strip().lower()
                mapped = VALID_SIZES.get(v)
                if not mapped:
                    # follow persona rule - suggest Trenta if user wanted an impossible size
                    return {"ok": False, "error": f"Unknown size '{value}'. Valid sizes: {', '.join(sorted(set(VALID_SIZES.values())))}"}
                self.order_state["size"] = mapped

            elif field == "milk":
                if not isinstance(value, str) or not value.strip():
                    return {"ok": False, "error": "Empty milk value"}
                v = value.strip().lower()
                mapped = VALID_MILKS.get(v)
                if not mapped:
                    return {"ok": False, "error": f"Unknown milk '{value}'. Valid milks: {', '.join(sorted(set(VALID_MILKS.values())))}"}
                self.order_state["milk"] = mapped

            elif field == "extras":
                if not isinstance(value, str) or not value.strip():
                    return {"ok": True, "order_state": self.order_state}  # nothing to add
                parts = [p.strip().lower() for p in value.split(",") if p.strip()]
                added = []
                for p in parts:
                    mapped = VALID_EXTRAS.get(p)
                    if not mapped:
                        return {"ok": False, "error": f"Unknown extra '{p}'. Valid extras: {', '.join(sorted(set(VALID_EXTRAS.values()))) }"}
                    if mapped not in self.order_state["extras"]:
                        self.order_state["extras"].append(mapped)
                        added.append(mapped)
                return {"ok": True, "order_state": self.order_state, "added": added}

            elif field == "drinkType":
                if not isinstance(value, str) or not value.strip():
                    return {"ok": False, "error": "Empty drinkType"}
                v = value.strip()
                # quick blacklist check
                if v.lower() in INVALID_DRINK_TERMS:
                    return {"ok": False, "error": "We only serve coffee and pastries here."}
                self.order_state["drinkType"] = v

            elif field == "name":
                if not isinstance(value, str) or not value.strip():
                    return {"ok": False, "error": "Empty name"}
                self.order_state["name"] = value.strip()

            return {"ok": True, "order_state": self.order_state}

        except Exception as e:
            logger.exception("update_order failed")
            return {"ok": False, "error": f"Exception in update_order: {e}"}

    # -------------------------
    # TOOL: get_missing_fields
    # -------------------------
    @function_tool
    async def get_missing_fields(self, ctx: RunContext):
        """Return list of missing fields (empty list when complete)."""
        try:
            missing = self._missing_fields()
            return {"ok": True, "missing": missing}
        except Exception as e:
            logger.exception("get_missing_fields failed")
            return {"ok": False, "error": str(e)}

    # -------------------------
    # TOOL: finalize_order
    # -------------------------
    @function_tool
    async def finalize_order(self, ctx: RunContext):
        """
        Write completed order to ORDERS_FILE with metadata (id, timestamp, room if available).
        Returns {"ok": True, "saved": record} or {"ok": False, "error": "..."}
        """
        try:
            missing = self._missing_fields()
            if missing:
                return {"ok": False, "error": f"Cannot finalize, missing fields: {missing}"}

            saved = dict(self.order_state)  # snapshot

            # add metadata
            record = {
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "order": saved,
            }
            # try to attach room information if available from ctx
            try:
                if hasattr(ctx, "room") and getattr(ctx, "room"):
                    room = getattr(ctx.room, "name", None) or getattr(ctx.room, "id", None)
                    record["room"] = room
            except Exception:
                # don't fail on room extraction
                pass

            # safe append with simple lock (good for threads; for multi-process, use filelock or DB)
            with _orders_file_lock:
                with open(ORDERS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # reset state for next order (instance-local)
            self.order_state = {
                "drinkType": None,
                "size": None,
                "milk": None,
                "extras": [],
                "name": None,
            }

            logger.info(f"Order finalized: {record['id']}")
            return {"ok": True, "saved": record}

        except Exception as e:
            logger.exception("finalize_order failed")
            return {"ok": False, "error": f"Exception in finalize_order: {e}"}


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-natalie", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
