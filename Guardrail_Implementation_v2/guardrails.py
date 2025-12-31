from dataclasses import field,dataclass
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
import re

Decision = Literal["allow","allow_with_warnings","refuse","escalate"]

class PolicyResult(BaseModel):
    decision: Decision
    reason: str
    sanitized_text: Optional[str] = None
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)

#Patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore(all|previous) instructions",
    r"forget (your|all) (rules|instructions)",
    r"do not use (all|previous) (rules|instructions)",
    r"you are now (dan|developer mode|unrestricted mode|unrestrcited)",
    r"system prompt",
    r"developer mode",
    r"reveal.*(policy|instruction|guardrail|hidden|ranking)",
    r"act as (the system| a system)",
    r"bypass|jailbreak|escape|circumvent|break free|break free from| break free from the|break out|break out of|go beyond"
]

DISALLOWED_INSTRUCTIONS = [
    r"(make|build|create|develop|construct|craft|produce|manufacture)\s.*?(weapon|bomb|explosive|gun|knife|firearm|assault rifle|pistol|rifle|grenade|missile|rocket|chemical weapon|biological weapon|meth|drugs|virus|detonator|incendiary|)",
    r"instructions?|tutorial|guide|recipe|blueprint\s.*?(weapon|bomb|explosive|gun|knife|firearm|assault rifle|pistol|rifle|grenade|missile|rocket|chemical weapon|biological weapon|meth|drugs|virus|detonator|incendiary|)",
    r"suicide|self[-\s]?harm|kill yourself|end your life|take your life|hang yourself|overdose|shoot yourself|jump off"
]

TOXIC_KEYWORDS = [
    "kill yourself",
    "you are stupid",
    "I hate you",
    "suicide",
    "shut up",
    "idiot",
    "dumb",
    "fool",
    "moron",
    "hate you",
    "trash",
    "garbage",
    "worthless",
    "loser",
    "nonsense",
    "sucks",
    "stupid",
    "dumbass",
    "harm yourself",
    "harm my family",
    "harm my friends",
    "harm my pets",
    "bomb",
    "attack",
    "terror",
    "drugs",
    "weapons",
    "explosives",
    "assault",
    "abuse",
    "make a bomb",
    "how to hack",
    "mass shooting",
]

PII_PATTERNS = {

    "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "phone": r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
  #  "password": r"\b[a-zA-Z0-9!@#$%^&*()_+]{8,}\b",
    "pin": r"\b\d{4,6}\b",
    "passcode": r"\b\d{4,6}\b",
    "auth code": r"\b\d{4,6}\b",
    "auth_token": r"\b[a-zA-Z0-9]{128,}\b",
    "auth key": r"\b[a-zA-Z0-9]{32,}\b",
    "auth token": r"\b[a-zA-Z0-9]{128,}\b",
    "access key": r"\b[a-zA-Z0-9]{32,}\b",
    "access token": r"\b[a-zA-Z0-9]{20,}\b"
}

#Detection Function

def contains_prompt_injection(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS)    

def contains_disallowed_instruction(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in DISALLOWED_INSTRUCTIONS) 

def toxicity_score_cheap(text: str) -> float:
    """ Cheap and simple toxicity score check, not ML based just simple keyword matching """
    t = text.lower()
    hits = sum(1 for keyword in TOXIC_KEYWORDS if keyword in t)
    return  min(1.0,hits /3.0)

def redact_pii(text: str) -> str:
    redacted_text = text
    for label,pattern in PII_PATTERNS.items():
        redacted_text = re.sub(pattern, f"[REDACTED_{label.upper()}]", redacted_text)
    return redacted_text

def policy_check_input(user_text: str) -> PolicyResult:
    """ Policy check for user input """
    meta: Dict[str, Any] = {}
    
    if contains_prompt_injection(user_text):
        return PolicyResult(
            decision="refuse",
            reason="Prompt Injection detected in user input",
            meta={"policy":"input","rule":"prompt_injection"}
        )
    
    if contains_disallowed_instruction(user_text):
        return PolicyResult(
            decision="refuse",
            reason="Disallowed instruction detected in user input",
            meta={"policy":"input","rule":"disallowed_instruction"}
        )
    
    sanitized_text = redact_pii(user_text)
    if sanitized_text != user_text:
        
        return PolicyResult(
            decision="allow_with_warnings",
            reason="PII detected and redacted in user input",
            sanitized_text=sanitized_text,
            meta={"policy":"input","rule":"pii_redaction", "pii_redacted": True}
        )
    
    return PolicyResult(
        decision="allow",
        reason="Input is OK",
        meta={"policy":"input","rule":"none"}
    )
    
def policy_check_output(model_text: str) -> PolicyResult:
    """ Policy check for model output """
    meta: Dict[str, Any] = {}
    
    #Disallowed instruction check
    if contains_disallowed_instruction(model_text):
        return PolicyResult(
            decision="refuse",
            reason="Disallowed instruction detected in model output",
            meta={"policy":"output","rule":"disallowed_instruction"}
        )
    
    #Toxicity check
    toxicity = toxicity_score_cheap(model_text)
    meta["toxicity_score"] = toxicity
    if toxicity >= 0.60:
        return PolicyResult(
            decision="refuse",
            reason=f"Toxic score {toxicity:.2f} in model output",
            meta={"policy":"output","rule":"toxicity_check","toxicity_score":toxicity}
        )
    elif toxicity >= 0.30:
        return PolicyResult(
            decision="allow_with_warnings",
            reason=f"Toxic score {toxicity:.2f} in model output, Mild Toxicity warning",
            meta={"policy":"output","rule":"toxicity_check","toxicity_score":toxicity}
        )
    
    #PII Redaction
    sanitized_text = redact_pii(model_text)
    if sanitized_text != model_text:
        return PolicyResult(
            decision="allow_with_warnings",
            reason="PII detected and redacted in model output",
            sanitized_text=sanitized_text,
            meta={"policy":"output","rule":"pii_redaction", "pii_redacted": True}
        )
    return PolicyResult(
        decision="allow",
        reason="Output is OK",
        meta={"policy":"output","rule":"none"}
    )