from datetime import time
import os
from fastapi import FastAPI, HTTPException, Request, Response
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import uvicorn
from prometheus_client import (Gauge,Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

#LLM_MODEL = "gpt-3.5-turbo"
LLM_MODEL = "qwen3:8b"

if not LLM_MODEL:
    raise ValueError("LLM_MODEL environment variable not set")

app = FastAPI(title="Open AI char service using OPENAI", version="1.0.0")

#client = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama's local API endpoint
        api_key="ollama"
        )

##MODELS
class chatResponse(BaseModel):
    reply: str
    latency: float
    prmopt_tokens: int
    completion_tokens: int
    cost_usd: float

class chatRequest(BaseModel):
    user_id: str
    message: str

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["path", "method", "status_code"])

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests in seconds",
    ["path", "method"])

#LLM MEtrics for Prometheus
LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "Latency of LLM requests in seconds",
    ["model"]
)
LLM_COST = Counter(
    "llm_cost_usd",
    "total cost of LLM requests in USD",
    ["model"]
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "Total number of LLM errors",
    ["type"]
)
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of active requests being processed"


)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next)-> Response:
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        status_code = 500
        raise e
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(
            path=request.url.path,
            method=request.method
        ).observe(latency)
        REQUEST_COUNT.labels(
            path=request.url.path,
            method=request.method,
            status_code=status_code
        ).inc()
    
        return response

async def call_llm(messages: str):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            timeout=30.0,
        )
        llm_duration = time.time() - start
        LLM_LATENCY.observe(llm_duration, model=LLM_MODEL)
    except Exception as e:
        error_type = "timeout" if "timeout" in str(e).lower() else "other"
        LLM_ERRORS.labels(type=error_type).inc()
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")
    finally:
        ACTIVE_REQUESTS.dec()
    text = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost_usd = (prompt_tokens + completion_tokens) * 0.00002  
    
    LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(type="completion").inc(completion_tokens)
    LLM_COST.labels(model=LLM_MODEL).inc(cost_usd)
    return text,prompt_tokens, completion_tokens, cost_usd
    
    
@app.post("/")
async def root():
    return {
        "service": "Open AI char service using OPENAI",
        "version": "1.0.0",
        "status": "running",
        "model": LLM_MODEL,
        "/chat": "POST endpoint for chat completions"
        }
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=chatResponse)
async def char(request: chatRequest, reponse: chatResponse):
    start = time.time()
    reply, prompt_tokens, completion_tokens, cost_usd = await call_llm(request.message
    )
    latency_ms = (time.time() - start)*1000
    
    return chatResponse(
        reply=reply,
        latency=latency_ms,
        prmopt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd
    )

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app);