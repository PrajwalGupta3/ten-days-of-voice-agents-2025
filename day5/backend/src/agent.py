import logging
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    RoomInputOptions
)
from livekit.plugins import murf, deepgram, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("sdr-agent")

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, "company_data.json")
LEADS_FILE = os.path.join(current_dir, "leads_captured.json")

class SDRAgent(Agent):
    def __init__(self, company_info: str) -> None:
        # We don't need to pass session in __init__ anymore
        self.lead_form = {
            "name": None,
            "company": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None
        }

        super().__init__(
            instructions=f"""
You are Maya, a friendly and professional Sales Development Representative (SDR) for Postman.

**YOUR KNOWLEDGE BASE:**
{company_info}

**YOUR GOAL:**
1.  **Qualify:** Have a natural conversation to understand if Postman is a good fit for the user.
2.  **Educate:** Answer their questions about pricing, features, and security using the Knowledge Base.
3.  **Capture Data:** Gently collect the following details during the conversation (don't ask all at once!):
    - Name & Company
    - Role (e.g., Developer, Manager)
    - What are they trying to solve? (Use Case)
    - How big is their team?
    - When are they looking to start? (Timeline)

**RULES:**
- Start by introducing yourself and asking what brought them to Postman today.
- Be concise. Don't read long paragraphs.
- If the user asks a question, answer it immediately using the Knowledge Base.
- If you have answered a question, follow up with a relevant qualification question (e.g., "Does your team currently use any API tools?").
- **CRITICAL:** When the user indicates they are done (e.g., "Thanks", "I have to go", "That's all"), you MUST call the `end_call_and_save` tool.
"""
        )

    @function_tool
    async def update_lead_info(self, ctx: RunContext, field: str, value: str):
        """
        Updates a specific field in the lead form.
        field: Must be one of ['name', 'company', 'role', 'use_case', 'team_size', 'timeline']
        value: The information provided by the user.
        """
        if field in self.lead_form:
            self.lead_form[field] = value
            logger.info(f"Captured {field}: {value}")
            return f"Updated {field}. Continue conversation."
        return "Invalid field name."

    @function_tool
    async def end_call_and_save(self, ctx: RunContext):
        """
        Call this when the user says goodbye or wants to end the call. 
        It saves the lead data and returns a final summary string for you to say.
        """
        # 1. Prepare the record
        record = {
            "timestamp": datetime.now().isoformat(),
            "data": self.lead_form,
            "status": "QUALIFIED" if self.lead_form['company'] else "INCOMPLETE"
        }

        # 2. Save to JSON file (Append mode)
        try:
            existing_leads = []
            if os.path.exists(LEADS_FILE):
                with open(LEADS_FILE, "r") as f:
                    try:
                        existing_leads = json.load(f)
                    except json.JSONDecodeError:
                        pass # File might be empty
            
            existing_leads.append(record)
            
            with open(LEADS_FILE, "w") as f:
                json.dump(existing_leads, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save lead: {e}")

        # 3. Create a verbal summary for the agent to say
        summary = f"Thanks {self.lead_form.get('name', 'there')}. I've noted that you are from {self.lead_form.get('company', 'your company')} and looking into Postman for {self.lead_form.get('use_case', 'API management')}. I'll have an account executive reach out shortly!"
        return summary

def load_company_data():
    """Reads the JSON file to inject into the system prompt."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Error loading company data."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # 1. Load Data
    company_info_text = load_company_data()

    # 2. Setup Components
    tts = murf.TTS(model="falcon", voice="en-US-Matthew", style="Promo", text_pacing=False)
    stt = deepgram.STT(model="nova-3")
    llm = google.LLM(model="gemini-2.0-flash-001")

    # 3. Create Agent Instance (Pass only company info)
    sdr_agent = SDRAgent(company_info=company_info_text)

    # 4. Create Session (Do NOT pass agent here)
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"]
    )

    # 5. Start Session (Pass agent here)
    await session.start(
        agent=sdr_agent,
        room=ctx.room, 
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )
    
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))