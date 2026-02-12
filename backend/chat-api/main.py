import os
from typing import Optional
import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import dotenv

dotenv.load_dotenv()

REDIS_URL = os.getenv('UPSTASH_REDIS_REST_URL')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Setup Redis and Gemini Client
r = redis.Redis.from_url(REDIS_URL)

# The client will automatically look for GEMINI_API_KEY in your environment variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class ChatRequest(BaseModel):
    user_message: str
    # Optional manual workload override from the user (1–9).
    user_workload: Optional[float] = None


DEFAULT_WORKLOAD = 5.0


def get_behavior_prompt(workload: float) -> str:
    """Specific instructions based on the neural workload score."""
    if workload >= 8.0:
        return "USER STATUS: CRITICAL LOAD. Action: Use extreme brevity. One-sentence answers only."
    elif workload >= 6.5:
        return "USER STATUS: HIGH LOAD. Action: Summarize. Use bullet points for all lists."
    else:
        return "USER STATUS: NORMAL. Action: Provide detailed, helpful, and conversational responses."


@app.post("/chat")
async def neuro_chat(request: ChatRequest):
    # 1) Base workload from Brain Engine (Redis)
    workload_from_engine = DEFAULT_WORKLOAD
    engine_value = None
    try:
        engine_value = r.get("latest_workload_score")
        if engine_value is not None:
            workload_from_engine = float(engine_value)
    except Exception:
        # If Redis is unavailable or has bad data, fall back to default
        workload_from_engine = DEFAULT_WORKLOAD

    # 2) If the user provides a workload (1–9), it overrides Brain Engine
    workload_source = "engine"
    if request.user_workload is not None:
        try:
            clamped = max(1.0, min(9.0, float(request.user_workload)))
            workload = clamped
            workload_source = "user"
        except (TypeError, ValueError):
            workload = workload_from_engine
            workload_source = "engine"
    else:
        workload = workload_from_engine
        workload_source = "engine" if engine_value is not None else "default"

    # 3) Generate behavior instruction
    behavior_instruction = get_behavior_prompt(workload)

    # 4) Call Gemini with System Instruction
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.user_message,
            config=types.GenerateContentConfig(
                system_instruction=behavior_instruction,
                temperature=0.7,
                # ADDED: Optional thinking budget for 2026 models to prevent truncation
                # thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        )
        ai_text = response.text
    except Exception as e:
        ai_text = f"Error: {str(e)}"

    return {
        "workload_detected": workload,
        "workload_source": workload_source,
        "ai_response": ai_text,
    }