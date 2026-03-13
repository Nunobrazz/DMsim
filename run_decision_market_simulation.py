"""
run_decision_market_simulation.py
──────────────────────────────────────────────────────────────────────
Runs the DAO governance simulation using the Decision Market (LMSR).
Each agent trades in the Action A and/or Action B market according to
their private beliefs (p_A, p_B) and their preferences (theta_A, theta_B).

Mechanism:
  - Two LMSR markets: one for Action A, one for Action B.
  - Each market tracks P(SUCCESS) and P(FAILURE) for that action.
  - Agent behaviour: they push each market's success probability toward
    their private belief. The magnitude of their trade is proportional
    to their theta (how much they care about that action).
  - Decision: the action whose market shows the highest P(SUCCESS) wins.
"""

import numpy as np
from profile_generator import generate_profiles, DAO_SCENARIO, build_matrix, Colors, display_profiles
from decision_market import DecisionMarket
from mbsr import buy_to_target_probability


# ── Configuration ─────────────────────────────────────────────────────────────

ACTIONS = ["Action A", "Action B"]
LIQUIDITY_B = 100.0   # LMSR liquidity parameter — larger = harder to move market
SUCCESS_OUTCOME = 0   # Outcome 0 = success, Outcome 1 = failure


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_decision_market_simulation():
    print(f"\n{Colors.BOLD}{Colors.CYAN}🚀 STARTING DECISION MARKET SIMULATION{Colors.END}")

    # 1. Generate agent profiles via LLM
    try:
        profiles = generate_profiles(DAO_SCENARIO, n_agents=5)
    except Exception as e:
        print(f"{Colors.RED}Profile generation failed: {e}{Colors.END}")
        return

    display_profiles(profiles)

    # 2. Initialise Decision Market
    dm = DecisionMarket(actions=ACTIONS, b=LIQUIDITY_B, n_outcomes=2)
    market_A = dm.get_market("Action A")
    market_B = dm.get_market("Action B")

    print(f"\n{Colors.BOLD}{Colors.HEADER}┌{'─'*88}┐{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}│ {'DECISION MARKET — AGENT TRADING':^86} │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}└{'─'*88}┘{Colors.END}")
    print(f"{'─'*40}")
    _initial_prices = dm.get_all_prices()
    print(f"{Colors.BLUE}Initial prices:  "
          f"A-SUCCESS={_initial_prices['Action A'][0]:.3f}  "
          f"B-SUCCESS={_initial_prices['Action B'][0]:.3f}{Colors.END}\n")

    # 3. Each agent trades to push markets toward their private beliefs
    # Strategy:
    #   - If agent believes p_action > current market price for success → buy SUCCESS shares
    #   - If agent believes p_action < current market price for success → buy FAILURE shares
    #   - Only trade in the market for the action they prefer (theta_X > 0) to avoid noise
    #     from agents who dislike both actions equally.

    print(f"{Colors.BOLD}{'Agent':<28} {'Mkt':<6} {'Direction':<16} {'p_belief':>9} {'p_before':>9} {'p_after':>9}{Colors.END}")
    print(f"{'─'*90}")

    for p in profiles:
        # --- Trade in Market A ---
        current_pA = market_A.get_current_price(SUCCESS_OUTCOME)
        if abs(p.p_A - current_pA) > 0.01:  # Only trade if meaningfully different
            if p.p_A > current_pA:
                buy_to_target_probability(market_A, SUCCESS_OUTCOME, min(p.p_A, 0.99))
                direction = f"{Colors.GREEN}BUY SUCCESS A{Colors.END}"
            else:
                buy_to_target_probability(market_A, 1, min(1.0 - p.p_A, 0.99))
                direction = f"{Colors.RED}BUY FAILURE A{Colors.END}"
            new_pA = market_A.get_current_price(SUCCESS_OUTCOME)
            print(f"{Colors.BOLD}{p.name[:28]:<28}{Colors.END} {'Mkt A':<6} {direction:<25} {p.p_A:>9.3f} {current_pA:>9.3f} {new_pA:>9.3f}")
        else:
            print(f"{Colors.BOLD}{p.name[:28]:<28}{Colors.END} {'Mkt A':<6} {Colors.YELLOW}NO TRADE{Colors.END}            {p.p_A:>9.3f} {current_pA:>9.3f} {current_pA:>9.3f}")

        # --- Trade in Market B ---
        current_pB = market_B.get_current_price(SUCCESS_OUTCOME)
        if abs(p.p_B - current_pB) > 0.01:
            if p.p_B > current_pB:
                buy_to_target_probability(market_B, SUCCESS_OUTCOME, min(p.p_B, 0.99))
                direction = f"{Colors.GREEN}BUY SUCCESS B{Colors.END}"
            else:
                buy_to_target_probability(market_B, 1, min(1.0 - p.p_B, 0.99))
                direction = f"{Colors.RED}BUY FAILURE B{Colors.END}"
            new_pB = market_B.get_current_price(SUCCESS_OUTCOME)
            print(f"  {'':26} {'Mkt B':<6} {direction:<25} {p.p_B:>9.3f} {current_pB:>9.3f} {new_pB:>9.3f}")
        else:
            print(f"  {'':26} {'Mkt B':<6} {Colors.YELLOW}NO TRADE{Colors.END}            {p.p_B:>9.3f} {current_pB:>9.3f} {current_pB:>9.3f}")
        print()

    # 4. Final market state and decision
    final_prices = dm.get_all_prices()
    pA_final = final_prices["Action A"][SUCCESS_OUTCOME]
    pB_final = final_prices["Action B"][SUCCESS_OUTCOME]
    recommended = dm.make_decision(SUCCESS_OUTCOME)

    print(f"\n{'─'*90}")
    print(f"{Colors.BOLD}FINAL CONSENSUS PROBABILITIES:{Colors.END}")
    print(f"  Action A  →  P(SUCCESS) = {Colors.BOLD}{Colors.CYAN}{pA_final:.4f}{Colors.END}")
    print(f"  Action B  →  P(SUCCESS) = {Colors.BOLD}{Colors.CYAN}{pB_final:.4f}{Colors.END}")

    print(f"\n{Colors.BOLD}{Colors.YELLOW}┌{'─'*48}┐{Colors.END}")
    decision_str = f"{recommended.upper()} APPROVED BY DECISION MARKET"
    print(f"{Colors.BOLD}{Colors.YELLOW}│ {decision_str:^46} │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}└{'─'*48}┘{Colors.END}")

    # 5. Show welfare at consensus probabilities
    print(f"\n{Colors.BLUE}Utilitarian Welfare at Consensus Probabilities:{Colors.END}")
    matrix = build_matrix(profiles)
    theta_A = matrix[:, 0]
    theta_B = matrix[:, 1]
    W_A = np.sum(theta_A * pA_final)
    W_B = np.sum(theta_B * pB_final)
    print(f"  W(A) = Σ θ_A · p_A* = {W_A:.2f}")
    print(f"  W(B) = Σ θ_B · p_B* = {W_B:.2f}")
    welfare_winner = "Action A" if W_A >= W_B else "Action B"
    match = "✓ Agrees" if welfare_winner == recommended else "✗ Disagrees"
    print(f"  Utilitarian winner: {welfare_winner}  |  Market decision: {recommended}  "
          f"→ {Colors.GREEN if '✓' in match else Colors.RED}{match}{Colors.END}\n")


if __name__ == "__main__":
    run_decision_market_simulation()
