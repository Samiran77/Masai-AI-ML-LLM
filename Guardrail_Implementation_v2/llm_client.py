from __future__ import annotations
import httpx
from typing import List, Any,Dict,Optional


class OllamaClient:
    """ A client for interacting with a Language Model (LLM) API."""
#    def __init__(self, base_url: str="http://localhost:11434", model: str="gemma3:1b"):
    def __init__(self, base_url: str="http://localhost:11434", model: str="sadiq-bd/llama3.2-1b-uncensored:latest"):
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json"
        }
    
    async def chat(self, message: List[Dict[str, str]], max_tokens: int=512, temperature: float=0.7) -> str:
        """ Send a chat message to the LLM and get the response. """
        payload:Dict[str,any] = {
            "model": self.model,
            "messages": message,
            "max_tokens": max_tokens,
            "options": {
                "temperature": temperature
            },
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
           resp = await client.post(f"{self.base_url}/api/chat",json=payload)
           resp.raise_for_status()
           data = resp.json()
           return data["message"]["content"]