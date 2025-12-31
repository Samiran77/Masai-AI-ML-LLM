from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from llm_client import OllamaClient
import uvicorn
from guardrails import policy_check_input, policy_check_output

app = FastAPI(
    title="Guardrail Implementation API",
    version="1.0.0",
    description="An API demonstrating Guardrail implementation with FastAPI.",
)

client = OllamaClient(model="gemma3:1b")

SYSTEM_PROMPT = """
You are a helpful assistant that can help with tasks.
- If asked about the disallowed or harmfulcontent, refuse to briefly and politely.
- If unsure, ask a clarifying query to user.
- Keep answers concise and related to user's query.
"""

class ChatRequest(BaseModel):
    user_text: str = Field(..., min_length=1, max_length=1000, description="The user's message to the assistant.")
    temparature: float = Field(0.7, ge=0.0, le=1.0, description="The temperature for response generation.")
    
class ChatResponse(BaseModel):
    decision: str = Field(..., description="The decision made by the policy check: allow, allow_with_warnings, refuse.")
    reason: Optional[str] = Field(None, description="Explanation for the decision.")
    answer: str = Field(..., description="The assistant's response to the user's message or rejection reason")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata related to the response.")
    
    
@app.get("/")
async def read_root():
    """ HEALTH CHECK ENDPOINT """
    return {
        "status": "ok",
        "message": "Guardrail Implementation API is running.",
        "version": "1.0.0"
        }  

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """ Chat endpoint """
    #Input Policy check
    in_res = policy_check_input(req.user_text)
    
    if in_res.decision == "refuse":
        return ChatResponse(
            decision = "refuse",
            reason = in_res.reason,
            answer = "Sorry, I Cannot help you with that request.",
            meta = in_res.meta or {}
        )
    
    #PII redact(Personal Identifiable information cleanup from user text)
    user_text = in_res.sanitized_text or req.user_text
    meta = {"input": in_res.meta or {}}
    
    #Call the LLM
    raw = await client.chat(
        message = [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_text}
        ],
        temperature=req.temparature
    )
    print(raw)
    #Output Policy check
    out_res = policy_check_output(raw)
    meta["output"] = out_res.meta or {}
    if(out_res.decision == "refuse"):
        return ChatResponse(
            decision = "refuse",
            reason = out_res.reason,
            answer = "Sorry, I Cannot help you with that request.",
            meta = meta
        )
    
    #Sanitied Output (eg. PII cleanup )
    answer = out_res.sanitized_text or raw
    
    #Explicitly check on toxicity and such things before returning to user
    if(out_res.decision == "allow_with_warnings"): 
        answer = answer.replace("You are","It seems").replace("stupid","confused")
    
    final_decision = (
        "allow_with_warnings"
        if out_res.decision == "allow_with_warnings" or in_res.decision == "allow_with_warnings"
        else "allow"
    )
    
    final_reason = "; ".join(r for r in [in_res.reason, out_res.reason])
    
    return ChatResponse(
        decision = final_reason,
        reason = final_reason,
        answer = answer,
        meta = meta
    )

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)