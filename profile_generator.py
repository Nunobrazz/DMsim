"""
profile_generator.py
──────────────────────────────────────────────────────────────────────
Generates [theta_A, theta_B, p_A, p_B] profiles using either Google Gemini
(free tier) or Groq (high-speed, no-credit-card free tier).

Usage Checklist:
1. Virtual Env: .venv/bin/pip install google-genai groq numpy
2. API Key (pick one):
   export GOOGLE_API_KEY="..."    # https://aistudio.google.com
   export GROQ_API_KEY="..."      # https://console.groq.com
3. Run:
   .venv/bin/python profile_generator.py
"""

import os
import json
import re
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np

# Providers
try:
    from google import genai
    from google.genai import types as gemini_types
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None


# ── Scenario (edit this variable to change the decision) ──────────────────────

DAO_SCENARIO = """
A lending protocol DAO must allocate its $2M growth budget for the next 12 months.
The council must choose between two competing treasury strategies.

  Action A: Aggressive Expansion — Launch $RISKY Collateral Market + Liquidity Mining
    - Onboard $RISKY (low-cap governance token, $4M market cap, 180% 90-day vol) as collateral at LTV 65%
    - Deploy $1.4M in liquidity mining rewards to bootstrap the new market
    - Reserve $600k for bad-debt coverage
    - Projected upside: 3x TVL growth to $255M if $RISKY market succeeds, protocol revenue jumps to $960k/month
    - Risk: oracle manipulation and cascade liquidation could cause insolvency;
      risk team estimates 30% probability of a severe bad-debt event within 12 months
    - Council members collectively hold ~8M $RISKY tokens (~$3M); success would 3-5x their positions

  Action B: Conservative Growth — Deepen Blue-Chip Markets + Build Insurance Fund
    - Add no new collateral types; focus on deepening ETH/wBTC/USDC markets
    - Deploy $800k in targeted incentives for large institutional depositors
    - Allocate $1.2M to a protocol insurance fund (bringing total reserves to $2.1M)
    - Projected upside: 40% TVL growth to $119M, protocol revenue rises to $450k/month
    - Risk: slower growth may lose market share to competitors;
      risk team estimates 95% probability of meeting targets, near-zero systemic risk
    - No direct financial benefit to $RISKY holders on the council

Protocol Stats:
  Total Value Locked: $85M | Bad debt reserve: $900k
  Monthly protocol revenue: $320k
  $RISKY token: current price $0.38, council members collectively hold ~8M tokens (~$3M)
"""


# ── Agent persona definitions ─────────────────────────────────────────────────
# Each agent has token holdings, epistemic priors, and a personality.
# NOTE: conflict of interest is a key dynamic — some agents benefit personally from the aggressive strategy.

DEFAULT_PERSONAS = [
    ("Conflicted Council Member",
     "You sit on the lending DAO council and personally hold 2.5M $RISKY tokens bought at $0.04. "
     "The aggressive expansion strategy (Action A) would bootstrap real utility for $RISKY and 3-5x "
     "your position. You genuinely believe the risk team is overly conservative and privately "
     "estimate a bad-debt event at only 10%. You see the conservative path as a missed opportunity."),

    ("Large Depositor (Lender)",
     "You have $1.2M in USDC deposited in the protocol earning 4.8% APY. You have zero $RISKY exposure. "
     "You've read the risk team's report and trust their 30% bad-debt estimate for the aggressive path. "
     "You see the insurance fund in Action B as directly protecting your capital. "
     "You acknowledge Action A's higher revenue could raise your yield, but insolvency risk terrifies you."),

    ("Neutral Risk Analyst",
     "You are a third-party auditor with no position in $RISKY and no deposits in this protocol. "
     "You've studied the oracle architecture and 90-day volatility. You estimate the bad-debt probability "
     "closer to 40% for Action A given current liquidity depth. You see Action B as financially sound "
     "but worry the protocol may lose competitive ground without bolder moves."),

    ("Small $RISKY Retail Holder",
     "You hold 15,000 $RISKY tokens bought at $0.25. You follow the project on Twitter and believe in "
     "its long-term roadmap. The aggressive strategy would pump $RISKY's price by 50%+. "
     "You think the risk team's report is FUD and put the bad-debt probability at 5%. "
     "You see Action B as the boring choice that leaves the protocol stagnating."),

    ("Protocol Insurance Fund Manager",
     "You manage the $900k bad-debt reserve. A single insolvency event from Action A could exceed "
     "the entire reserve. You hold no $RISKY. Action B would more than double your reserve to $2.1M, "
     "giving the protocol a real safety net for the first time. You fully trust the 30% bad-debt estimate "
     "for Action A and believe even that may be understated given correlation risk."),
]


# ── Prompt Template ───────────────────────────────────────────────────────────

_BATCH_PROMPT_TEMPLATE = """\
You are simulating a DAO governance vote. The specific decision context is:

{context}

Generate exactly {n} agent profiles. For each agent:
1. Adopt their character fully (their token holdings, financial stake, knowledge level)
2. Reason from their perspective about which action benefits them
3. Then output their calibrated numbers

Return ONLY this JSON object (no markdown, no extra text):
{{
  "profiles": [
    {{
      "name": "<agent role>",
      "theta_A": <float in [-50, 50]; idiosyncratic preference for Action A>,
      "theta_B": <float in [-50, 50]; idiosyncratic preference for Action B>,
      "p_A": <float in (0,1); this agent's personal belief that Action A succeeds>,
      "p_B": <float in (0,1); this agent's personal belief that Action B succeeds>,
      "rationale": "<1-2 sentences in-character, explaining the numbers>"
    }}
  ]
}}

Hard constraints:
- theta values MUST be strictly within [-50.0, 50.0]
- p_A and p_B are INDEPENDENT beliefs (do not need to sum to 1)
- Agent priors must be heterogeneous; make them genuinely disagree
- None of the agents sure of anything. So p_A and p_B should be in (0.1, 0.9)
"""



@dataclass
class AgentProfile:
    name: str
    theta_A: float
    theta_B: float
    p_A: float
    p_B: float
    rationale: str

    def to_list(self) -> list[float]:
        return [self.theta_A, self.theta_B, self.p_A, self.p_B]


# ── Provider Clients ──────────────────────────────────────────────────────────

class LLMProvider:
    def generate(self, prompt: str, model: str) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        if not genai:
            raise ImportError("google-genai not installed.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, model: str) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=gemini_types.GenerateContentConfig(temperature=0.7)
        )
        return response.text

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str):
        if not Groq:
            raise ImportError("groq not installed.")
        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, model: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content


# ── Utilities ─────────────────────────────────────────────────────────────────

def retry_with_backoff(func, max_retries=3, initial_delay=2):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "limit" in str(e).lower():
                    if i == max_retries - 1: raise e
                    time.sleep(delay + random.uniform(0, 1))
                    delay *= 2
                else: raise e
    return wrapper

def _parse_json(text: str) -> dict:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()
    # Find the first { and last } to be safe
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)


# ── Core Functions ────────────────────────────────────────────────────────────

def get_provider() -> tuple[LLMProvider, str]:
    """Auto-detects provider based on environment variables."""
    if os.environ.get("GROQ_API_KEY"):
        print("  [Using Groq Provider]")
        return GroqProvider(os.environ["GROQ_API_KEY"]), "llama-3.3-70b-versatile"
    elif os.environ.get("GOOGLE_API_KEY"):
        print("  [Using Gemini Provider]")
        return GeminiProvider(os.environ["GOOGLE_API_KEY"]), "gemini-1.5-flash"
    else:
        raise ValueError("Set either GROQ_API_KEY or GOOGLE_API_KEY.")

def generate_profiles(context: str, n_agents: int = 5) -> list[AgentProfile]:
    """Generates all profiles in a single batch call."""
    provider, model = get_provider()
    prompt = _BATCH_PROMPT_TEMPLATE.format(context=context, n=n_agents)
    
    print(f"  → Generating {n_agents} profiles via {model}...", end=" ", flush=True)
    
    @retry_with_backoff
    def _do_call():
        return provider.generate(prompt, model)
    
    raw = _do_call()
    data = _parse_json(raw)
    
    profiles = [
        AgentProfile(
            name=p["name"],
            theta_A=float(p["theta_A"]),
            theta_B=float(p["theta_B"]),
            p_A=float(p["p_A"]),
            p_B=float(p["p_B"]),
            rationale=p.get("rationale", "")
        )
        for p in data["profiles"]
    ]
    print("done")
    return profiles


# ── Display Utilities ──────────────────────────────────────────────────────────

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def build_matrix(profiles: list[AgentProfile]) -> np.ndarray:
    return np.array([p.to_list() for p in profiles])

def display_profiles(profiles: list[AgentProfile]) -> None:
    """Prints a professional, color-coded report of the generated agents."""
    if not profiles:
        print(f"{Colors.RED}No profiles generated.{Colors.END}")
        return

    # Print Decision Context Summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌{'─'*88}┐{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}│ {'VOTING SCENARIO & DECISION CONTEXT':^86} │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─'*88}┘{Colors.END}")
    
    # Strip unnecessary whitespace and print scenario
    scenario_clean = DAO_SCENARIO.strip()
    for line in scenario_clean.split('\n'):
        print(f"  {line}")

    print(f"\n{Colors.BOLD}{Colors.HEADER}┌{'─'*88}┐{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}│ {'STRATEGIC AGENT PROFILES':^86} │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}└{'─'*88}┘{Colors.END}")

    print(f"{Colors.BOLD}{'Agent Persona':<30} │ {'θ_A':>8} │ {'θ_B':>8} │ {'p_A':>8} │ {'p_B':>8}{Colors.END}")
    print(f"{'─'*30}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}")

    for p in profiles:
        tA_col = Colors.GREEN if p.theta_A > 0 else (Colors.RED if p.theta_A < 0 else "")
        tB_col = Colors.GREEN if p.theta_B > 0 else (Colors.RED if p.theta_B < 0 else "")
        pA_col = Colors.CYAN if p.p_A > 0.5 else Colors.YELLOW
        pB_col = Colors.CYAN if p.p_B > 0.5 else Colors.YELLOW

        print(f"{Colors.BOLD}{p.name:<30}{Colors.END} │ "
              f"{tA_col}{p.theta_A:>8.1f}{Colors.END} │ "
              f"{tB_col}{p.theta_B:>8.1f}{Colors.END} │ "
              f"{pA_col}{p.p_A:>8.3f}{Colors.END} │ "
              f"{pB_col}{p.p_B:>8.3f}{Colors.END}")
        print(f"  {Colors.BLUE}↳ {p.rationale}{Colors.END}\n")

    matrix = build_matrix(profiles)
    means = np.mean(matrix, axis=0)
    
    print(f"{'─'*89}")
    print(f"{Colors.BOLD}{'MARKET CONSENSUS (MEAN)':<30}{Colors.END} │ "
          f"{means[0]:>8.1f} │ {means[1]:>8.1f} │ {Colors.BOLD}{means[2]:>8.3f}{Colors.END} │ {Colors.BOLD}{means[3]:>8.3f}{Colors.END}")
    print(f"{'─'*89}\n")


if __name__ == "__main__":
    try:
        print(f"\n{Colors.CYAN}Initializing Portfolio Strategy Generation...{Colors.END}")
        profiles = generate_profiles(DAO_SCENARIO)
        display_profiles(profiles)

        matrix = build_matrix(profiles)
        print(f"{Colors.GREEN}✓ Matrix generated.{Colors.END} Input for Organization established.")
        print(f"{Colors.BLUE}Profile Matrix Shape: {matrix.shape}{Colors.END}\n")

    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}ERROR:{Colors.END} {e}")
        print(f"\n{Colors.YELLOW}Checklist:{Colors.END}")
        print(" 1. Ensure GROQ_API_KEY or GOOGLE_API_KEY is exported.")
        print(" 2. Verify network connectivity.")
