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
| `run_vcgr_simulation.py` | End-to-end VCG simulation runner |
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

### 1. Environment Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install numpy

# For LLM-powered profile generation, also install:
pip install google-genai groq python-dotenv
```

### 2. API Keys

Ensure either `GOOGLE_API_KEY` or `GROQ_API_KEY` is exported in your environment:

```bash
export GOOGLE_API_KEY="..."  # From https://aistudio.google.com
# OR
export GROQ_API_KEY="..."    # From https://console.groq.com
```

---

### Interactive CLI

Both `LMSR` and `DecisionMarket` provide a live trading session:

```bash
python mbsr.py            # Interactive single LMSR market
python decision_market.py # Interactive multi-action decision market
```

### Running Simulations

These scripts run full end-to-end simulations with LLM-generated agent profiles:

```bash
# Run the VCGR (VCG-inspired) simulation
python run_vcgr_simulation.py

# Run the Decision Market (Conditional LMSR) simulation
python run_decision_market_simulation.py
```

### Customizing the Simulation

You can easily modify the simulation parameters, decision context, and agent archetypes:

1.  **Edit the Decision Context**: Open `profile_generator.py` and modify the `DAO_SCENARIO` string. This defines the overall situation, the actions available (Action A vs Action B), and the technical or financial stakes.
2.  **Edit Agent Personas**: While the LLM generates specific agent details, you can influence their behavior by modifying the `_BATCH_PROMPT_TEMPLATE` in `profile_generator.py` or by adding specific persona requirements to the `DAO_SCENARIO` text.
3.  **Adjust Simulation Scale**: In `run_vcgr_simulation.py` or `run_decision_market_simulation.py`, you can change the `n_agents` parameter in the `generate_profiles` call to simulate more or fewer participants.

---

```
DMsim/
├── mbsr.py                           # LMSR market core
├── vcgr.py                           # VCGR decision mechanism
├── organization.py                   # Agent organization layer
├── decision_market.py                # Multi-action conditional Decision Market
├── profile_generator.py              # LLM-based agent profile generator
├── run_vcgr_simulation.py             # VCGR end-to-end simulation
└── run_decision_market_simulation.py  # Decision Market simulation
```

---

| Package | Purpose |
|---|---|
| `numpy` | Array operations and vectorized math |
| `google-genai` | Google Gemini API client |
| `groq` | Groq API client |
| `python-dotenv` | API key management via `.env` file |

---

## License

This project is released for academic and research purposes.