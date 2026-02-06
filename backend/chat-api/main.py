import os
import redis
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# 1. Setup Redis and Gemini Client
r = redis.Redis(host='cache', port=6379, decode_responses=True)

# The client will automatically look for GEMINI_API_KEY in your environment variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    user_message: str

DEFAULT_WORKLOAD = 5.0

def get_behavior_prompt(workload):
    """Specific instructions based on the neural workload score."""
    if workload >= 8.0:
        return "USER STATUS: CRITICAL LOAD. Action: Use extreme brevity. One-sentence answers only."
    elif workload >= 6.5:
        return "USER STATUS: HIGH LOAD. Action: Summarize. Use bullet points for all lists."
    else:
        return "USER STATUS: NORMAL. Action: Provide detailed, helpful, and conversational responses."

@app.post("/chat")
async def neuro_chat(request: ChatRequest):
    # Fetch workload from Service 2 (Brain Engine) via Redis
    try:
        raw_score = r.get("latest_workload_score")
        workload = float(raw_score) if raw_score else DEFAULT_WORKLOAD
    except:
        workload = DEFAULT_WORKLOAD

    # Generate behavior instruction
    behavior_instruction = get_behavior_prompt(workload)

    # Call Gemini with System Instruction
    # We use Flash for lower latency in a real-time demo
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
        "ai_response": ai_text
    }