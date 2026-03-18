"""
api.py
──────────────────────────────────────────────────────────────────────
FastAPI backend for the DMsim frontend.
Wraps profile_generator.py to expose profile generation via HTTP.
"""

from dotenv import load_dotenv
load_dotenv("keys.env")

import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from profile_generator import (
    generate_profiles,
    AgentProfile,
    DAO_SCENARIO,
    DEFAULT_PERSONAS,
    _BATCH_PROMPT_TEMPLATE,
    get_provider,
    retry_with_backoff,
    _parse_json,
)
from vcgr import DecisionMechanism

app = FastAPI(title="DMsim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Daily Request Rate Limiter ───────────────────────────────────────────────

DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "100"))

_rate_state = {"count": 0, "reset_at": time.time() + 86400}


def _check_daily_limit():
    """Raise 429 if the daily LLM request budget is exhausted."""
    now = time.time()
    if now >= _rate_state["reset_at"]:
        _rate_state["count"] = 0
        _rate_state["reset_at"] = now + 86400
    if _rate_state["count"] >= DAILY_REQUEST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily request limit ({DAILY_REQUEST_LIMIT}) reached. Try again tomorrow.",
        )
    _rate_state["count"] += 1


# ── Request / Response Models ────────────────────────────────────────────────

class PersonaInput(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class GenerateRequest(BaseModel):
    context: str = Field(..., min_length=1, description="The scenario/context description")
    action_a: str = Field(..., min_length=1, description="Description of Action A")
    action_b: str = Field(..., min_length=1, description="Description of Action B")
    personas: List[PersonaInput] = Field(..., min_length=1)


class ProfileResponse(BaseModel):
    name: str
    theta_A: float
    theta_B: float
    p_A: float
    p_B: float
    rationale: str


class GenerateResponse(BaseModel):
    profiles: List[ProfileResponse]


class DefaultsResponse(BaseModel):
    context: str
    action_a: str
    action_b: str
    personas: List[PersonaInput]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/defaults", response_model=DefaultsResponse)
def get_defaults():
    """Return the default scenario, actions, and personas."""
    lines = DAO_SCENARIO.strip().split("\n")
    # Extract the top-level context (first paragraph before Action A)
    context_lines = []
    action_a_lines = []
    action_b_lines = []
    current = "context"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Action A:"):
            current = "a"
            action_a_lines.append(stripped.replace("Action A:", "").strip())
        elif stripped.startswith("Action B:"):
            current = "b"
            action_b_lines.append(stripped.replace("Action B:", "").strip())
        elif stripped.startswith("Protocol Stats:"):
            current = "context"
            context_lines.append(line)
        elif current == "context":
            context_lines.append(line)
        elif current == "a":
            action_a_lines.append(stripped)
        elif current == "b":
            action_b_lines.append(stripped)

    return DefaultsResponse(
        context="\n".join(context_lines).strip(),
        action_a="\n".join(action_a_lines).strip(),
        action_b="\n".join(action_b_lines).strip(),
        personas=[
            PersonaInput(name=name, description=desc)
            for name, desc in DEFAULT_PERSONAS
        ],
    )


@app.post("/api/generate-profiles", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate agent profiles using the LLM."""
    _check_daily_limit()
    # Build the full context string combining scenario + actions + personas
    persona_block = "\n".join(
        f"  Agent {i+1}: {p.name}\n    {p.description}"
        for i, p in enumerate(req.personas)
    )

    full_context = f"""{req.context}

  Action A: {req.action_a}

  Action B: {req.action_b}

Agents participating in the vote:
{persona_block}
"""

    prompt = _BATCH_PROMPT_TEMPLATE.format(context=full_context, n=len(req.personas))

    try:
        provider, model = get_provider()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    @retry_with_backoff
    def _call():
        return provider.generate(prompt, model)

    try:
        raw = _call()
        data = _parse_json(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    try:
        profiles = []
        for p in data["profiles"]:
            profiles.append(ProfileResponse(
                name=p["name"],
                theta_A=float(p["theta_A"]),
                theta_B=float(p["theta_B"]),
                p_A=float(p["p_A"]),
                p_B=float(p["p_B"]),
                rationale=p.get("rationale", ""),
            ))
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"LLM returned malformed profiles: {e}")

    return GenerateResponse(profiles=profiles)


# ── VCGR Simulation ─────────────────────────────────────────────────────────

class VCGRProfileInput(BaseModel):
    name: str
    theta_A: float
    theta_B: float
    p_A: float
    p_B: float


class VCGRRequest(BaseModel):
    profiles: List[VCGRProfileInput]
    budget: float = Field(5.0, gt=0, description="Budget constraint (c)")
    delta: float = Field(1.0, description="Realized outcome metric (delta)")


class VCGRAgentResult(BaseModel):
    name: str
    optimal_report: float
    transfer: float
    reward: float
    payoff: float


class VCGRResponse(BaseModel):
    allocation: int
    sum_reports: float
    budget: float
    delta: float
    agents: List[VCGRAgentResult]


@app.post("/api/vcgr-simulate", response_model=VCGRResponse)
def vcgr_simulate(req: VCGRRequest):
    """Run VCGR mechanism with optimal (strategic) reports derived from agent profiles."""
    n = len(req.profiles)
    budget = req.budget

    # Compute optimal reports for each agent
    initial_reports = [0.0] * n
    game = DecisionMechanism(initial_reports, budget)

    optimal_reports = []
    for p in req.profiles:
        try:
            report = game.optimal_report([p.theta_A, p.theta_B, p.p_A, p.p_B])
        except ValueError:
            report = 0.0
        optimal_reports.append(report)

    # Create a new game with the optimal reports
    game = DecisionMechanism(optimal_reports, budget)
    allocation, t, r, pi = game.resolve_game(req.delta)

    agents = []
    for i, p in enumerate(req.profiles):
        agents.append(VCGRAgentResult(
            name=p.name,
            optimal_report=round(optimal_reports[i], 4),
            transfer=round(float(t[i]), 4),
            reward=round(float(r[i]), 4),
            payoff=round(float(pi[i]), 4),
        ))

    return VCGRResponse(
        allocation=int(allocation),
        sum_reports=round(sum(optimal_reports), 4),
        budget=budget,
        delta=req.delta,
        agents=agents,
    )
