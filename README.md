# MBSR — Market-Based Social Choice & Resource Allocation

A research-oriented Python framework for mechanism design and collective decision-making. It combines two foundational approaches — **Logarithmic Market Scoring Rule (LMSR)** prediction markets and a **VCG-inspired reporting mechanism (VCGR)** — into an integrated toolkit for organizations facing binary decisions under uncertainty.

---

## Overview

The framework models an organization of $n$ agents, each holding private beliefs and idiosyncratic preferences over two candidate outcomes $A$ and $B$. It provides:

| Module | Description |
|---|---|
| `mbsr.py` | Core LMSR prediction market (AMM) with interactive CLI |
| `vcgr.py` | VCGR decision mechanism — report, allocate, transfer, and resolve |
| `organization.py` | Agent organization layer, bridging LMSR and VCGR |
| `decision_market.py` | Multi-action conditional Decision Market built on LMSR |
| `profile_generator.py` | LLM-powered diverse agent profile generator |
| `run_simulation.py` | End-to-end VCG + LMSR simulation runner |
| `run_decision_market_simulation.py` | Full Decision Market simulation runner |

---

## The Mathematics

### 1. LMSR — Logarithmic Market Scoring Rule

#### Cost Function

The LMSR automated market maker prices a set of outcomes using the cost function:

$$C(\mathbf{q}) = b \ln\left(\sum_{i} e^{q_i / b}\right)$$

where $q_i$ is the number of shares outstanding for outcome $i$, and $b > 0$ is the **liquidity parameter** controlling how quickly prices move.

#### Outcome Probabilities

The implicit price (i.e., probability) of outcome $i$ is the partial derivative of the cost function with respect to $q_i$:

$$P_i = \frac{\partial C}{\partial q_i} = \frac{e^{q_i / b}}{\displaystyle\sum_{j} e^{q_j / b}}$$

#### Transaction Cost

The cost of a trade that changes the shares of outcome $i$ by $\Delta q_i$ is:

$$\text{cost}(\Delta q_i) = C(\mathbf{q} + \Delta \mathbf{e}_i \cdot \Delta q_i) - C(\mathbf{q})$$

#### Maximum Market Maker Loss

For a binary market initialized at $\mathbf{q} = \mathbf{0}$, the worst-case loss for the market maker is bounded by:

$$\text{Loss}_{\max} = b \ln(2)$$

#### Targeting a Probability — Closed-Form Solution

To derive the exact number of shares $\Delta q_i$ required to move the price of outcome $i$ to a target $P_{\text{target}}$, define the sum of exponents for all *other* outcomes:

$$S_{-i} = \sum_{j \neq i} e^{q_j / b}$$

Setting the new price equal to the target:

$$\frac{e^{(q_i + \Delta q_i)/b}}{e^{(q_i + \Delta q_i)/b} + S_{-i}} = P_{\text{target}}$$

Solving for $\Delta q_i$:

$$\boxed{\Delta q_i = b \ln\!\left(\frac{P_{\text{target}}}{1 - P_{\text{target}}} \cdot S_{-i}\right) - q_i}$$

#### Shares from a Budget

Given a fixed budget $y$ and current price $p_i$, the exact number of shares purchasable is:

$$x = b \ln\!\left(\frac{e^{y/b} - 1 + p_i}{p_i}\right)$$

---

### 2. VCGR — VCG-Inspired Reporting Mechanism

Each agent $i$ reports a scalar $m_i \in \mathbb{R}$ representing their net preference for action $A$ over $B$.

#### Allocation Rule

The action $a^* \in \{A, B\}$ is chosen based on the aggregate report:

$$a^* = \begin{cases} A & \text{if } M = \displaystyle\sum_{i=1}^n m_i \geq 0 \\ B & \text{otherwise} \end{cases}$$

#### Transfer Rule (Pivot Mechanism)

Let $S_{-i} = M - m_i$ be the sum of all reports excluding agent $i$. The transfer to agent $i$ is:

$$t_i = \begin{cases} -S_{-i} & \text{if } -m_i > S_{-i} > 0 \quad \text{(agent $i$ is pivotal for $A$)} \\ S_{-i} & \text{if } -m_i < S_{-i} < 0 \quad \text{(agent $i$ is pivotal for $B$)} \\ 0 & \text{otherwise} \end{cases}$$

#### Reward Rule

After the outcome $\Delta \in \mathbb{R}$ is realized (positive = action was beneficial):

$$r_i = \begin{cases} -t_i + \dfrac{c}{n} & \text{if } \Delta > 0 \\ t_i - \dfrac{c}{n} & \text{if } \Delta \leq 0 \end{cases}$$

where $c$ is the budget constraint and $n$ is the number of agents.

#### Final Payoff

$$\pi_i = t_i + r_i$$

#### Optimal Report (Dominant Strategy)

Given agent $i$'s type $(\theta_i^A, \theta_i^B, p_i^A, p_i^B)$, the dominant strategy report is:

$$m_i^* = \frac{\theta_i^A - \theta_i^B}{2(1 - p_i^*)} + \frac{c}{c} \cdot \frac{p_i^A - p_i^B}{1 - p_i^*}$$

where $p_i^* = p_i^A$ if $\theta_i^A > \theta_i^B$, and $p_i^* = p_i^B$ otherwise.

---

### 3. Agent Type and Expected Utility

Each agent $i$ is characterized by a **type profile** $(\theta_i^A,\, \theta_i^B,\, p_i^A,\, p_i^B)$:

| Parameter | Description |
|---|---|
| $\theta_i^A$ | Idiosyncratic monetary value if action $A$ succeeds |
| $\theta_i^B$ | Idiosyncratic monetary value if action $B$ succeeds |
| $p_i^A$ | Agent $i$'s private belief that $A$ succeeds |
| $p_i^B$ | Agent $i$'s private belief that $B$ succeeds |

The expected utility for each action is:

$$EU_i^A = \theta_i^A \cdot p_i^A, \qquad EU_i^B = \theta_i^B \cdot p_i^B$$

The VCG report collapses these into a single scalar:

$$m_i = EU_i^A - EU_i^B$$

---

### 4. Decision Market (Conditional LMSR)

A Decision Market maintains one independent LMSR market per candidate action. Agents trade in the market(s) they have beliefs about. The recommended action $a^*$ is:

$$a^* = \arg\max_{a \in \mathcal{A}} \; P_{\text{success}}^a$$

where $P_{\text{success}}^a$ is the market-implied probability of success *given* that action $a$ is taken.

---

## Installation

```bash
pip install numpy
```

For LLM-powered profile generation (`profile_generator.py`), also install:

```bash
pip install openai python-dotenv
```

---

## Quick Start

### LMSR Prediction Market

```python
from mbsr import LMSR, buy_to_target_probability
import numpy as np

market = LMSR(event="Invest in Project X?", q=[0, 0], b=100, market_maker_fee=0)

# Current prices (start at 50/50)
print(market.get_current_price(0))  # 0.5

# Buy shares and observe price movement
market.buy_shares(outcome=0, shares=50)
print(market.get_current_price(0))  # > 0.5

# Move the market to exactly 75% probability for outcome 0
shares_bought, cost = buy_to_target_probability(market, outcome=0, target_probability=0.75)
```

### VCGR Decision Mechanism

```python
from vcgr import DecisionMechanism

reports = [10.0, -4.0, -8.0]
game = DecisionMechanism(reports=reports, budget_constraint=5.0)

print(game.get_allocation())     # True (A wins) or False (B wins)
print(game.calculate_t())        # Transfer array for all agents
game.display_summary(delta=5.0)  # Full resolution with payoffs
```

### Organization with LMSR + VCGR Integration

```python
from organization import Organization
from vcgr import DecisionMechanism
from mbsr import LMSR, buy_to_target_probability

# [theta_A, theta_B, p_A, p_B] for each agent
agent_data = [
    [100.0,  50.0, 0.6, 0.3],
    [-20.0, 150.0, 0.4, 0.5],
    [ 10.0,  10.0, 0.3, 0.5],
    [ 80.0, -10.0, 0.7, 0.4],
]

org = Organization(agent_data)

# Step 1: Run VCG with private beliefs
reports = org.get_vcg_reports()
vcgr = DecisionMechanism(reports, budget_constraint=100.0)
vcgr.display_summary(delta=1.0)

# Step 2: Run LMSR to aggregate beliefs into consensus probabilities
market = LMSR("Org Decision", [0, 0], b=100.0, market_maker_fee=0.0)
for i in range(org.n):
    buy_to_target_probability(market, outcome=0, target_probability=org.p_A[i])

consensus_p_A = market.get_current_price(0)

# Step 3: Re-run VCG with consensus beliefs
reports_consensus = org.get_vcg_reports(p_A_override=consensus_p_A)
vcgr_consensus = DecisionMechanism(reports_consensus, budget_constraint=100.0)
vcgr_consensus.display_summary(delta=1.0)
```

### Decision Market (Multi-Action)

```python
from decision_market import DecisionMarket
from mbsr import buy_to_target_probability

dm = DecisionMarket(actions=["Invest in AI", "Invest in Biotech"], b=100.0)

# Agents trade to express their conditional beliefs
buy_to_target_probability(dm.get_market("Invest in AI"),     outcome=0, target_probability=0.80)
buy_to_target_probability(dm.get_market("Invest in Biotech"), outcome=0, target_probability=0.70)

dm.display_summary()
print("Recommended action:", dm.make_decision())
```

### Interactive CLI

Both `LMSR` and `DecisionMarket` provide a live trading session:

```bash
python mbsr.py          # Interactive single LMSR market
python decision_market.py  # Interactive multi-action decision market
```

---

## Repository Structure

```
mbsr/
├── mbsr.py                          # LMSR market core
├── vcgr.py                          # VCGR decision mechanism
├── organization.py                  # Agent organization layer
├── decision_market.py               # Multi-action conditional Decision Market
├── profile_generator.py             # LLM-based agent profile generator
├── run_simulation.py                # VCG + LMSR end-to-end simulation
└── run_decision_market_simulation.py # Decision Market simulation
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations and vectorized math |
| `openai` | LLM-powered profile generation (optional) |
| `python-dotenv` | API key management via `.env` file (optional) |

---

## License

This project is released for academic and research purposes.