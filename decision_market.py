import numpy as np
from mbsr import LMSR, buy_to_target_probability, lmsr_prices, lmsr_total_value


class DecisionMarket:
    """
    A Decision Market consists of one conditional LMSR prediction market per
    action. Agents trade in the market(s) they believe in, and the recommended
    action is the one whose market assigns the highest probability to the
    desired (success) outcome.

    Outcome convention:
        outcome 0 = SUCCESS / desired result
        outcome 1 = FAILURE / undesired result
        (can be extended to more outcomes)
    """

    def __init__(self, actions, b=100.0, n_outcomes=2, market_maker_fee=0.0):
        """
        Initialises one LMSR market per action.

        Parameters:
        actions (list of str): The candidate actions/decisions (e.g. ['Invest', 'Wait']).
        b (float): Liquidity parameter shared across all markets.
        n_outcomes (int): Number of outcomes per market (default 2: success / failure).
        market_maker_fee (float): Fee per transaction.
        """
        self.actions = actions
        self.n_outcomes = n_outcomes
        self.b = float(b)
        self.market_maker_fee = market_maker_fee

        # One LMSR market per action, initialised with equal shares
        self.markets = {
            action: LMSR(
                event=f"Outcome if '{action}'",
                q=[0.0] * n_outcomes,
                b=b,
                market_maker_fee=market_maker_fee,
            )
            for action in actions
        }

    def get_market(self, action):
        """Returns the LMSR market for the given action."""
        if action not in self.markets:
            raise KeyError(f"Action '{action}' not found. Available: {self.actions}")
        return self.markets[action]

    def get_all_prices(self):
        """
        Returns a dict mapping each action to its current outcome prices.

        Returns:
        dict: { action -> np.ndarray of prices }
        """
        return {
            action: lmsr_prices(mkt.shares, mkt.b)
            for action, mkt in self.markets.items()
        }

    def make_decision(self, success_outcome=0):
        """
        Picks the action whose conditional market forecasts the highest
        probability for the success outcome (outcome 0 by default).

        Parameters:
        success_outcome (int): The outcome index considered 'success'.

        Returns:
        str: The recommended action.
        """
        prices = self.get_all_prices()
        best_action = max(prices, key=lambda a: prices[a][success_outcome])
        return best_action

    def display_summary(self, success_outcome=0):
        """
        Prints the current state of every conditional market.
        """
        all_prices = self.get_all_prices()
        recommended = self.make_decision(success_outcome)

        print("=== Decision Market Summary ===")
        print(f"  Actions: {self.actions}")
        print(f"  Liquidity (b): {self.b} | Outcomes: {self.n_outcomes}\n")

        for action, prices in all_prices.items():
            tag = " <<< RECOMMENDED" if action == recommended else ""
            print(f"  Action '{action}':{tag}")
            for i, p in enumerate(prices):
                label = "SUCCESS" if i == success_outcome else f"Outcome {i}"
                print(f"    {label}: {p:.4f}")
            mkt = self.markets[action]
            print(f"    Market Value (V): {mkt.get_market_value():.4f}")
            print()


    def interactive_session(self, success_outcome=0):
        """ 
        CLI session that lets agents trade in any of the conditional markets.

        Commands:
            prices                                - Show all market prices.
            status                                - Full market summary.
            buy <action> <outcome> <shares>       - Buy shares.
            sell <action> <outcome> <shares>      - Sell shares.
            target <action> <outcome> <prob>      - Buy to target probability.
            budget <action> <outcome> <amount> [buy] - Shares from budget.
            decide                                - Show current recommended decision.
            quit                                  - Exit session.
        """
        print("\n=== Decision Market Interactive Session ===")
        print(f"  Actions: {self.actions}")
        print("  Commands: buy | sell | target | budget | prices | status | decide | quit\n")
        self.display_summary(success_outcome)

        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Session ended.")
                break

            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()

            if cmd == "quit":
                print("  Session ended.")
                break

            elif cmd == "prices":
                all_prices = self.get_all_prices()
                for action, prices in all_prices.items():
                    print(f"  [{action}] " + "  ".join(f"Outcome {i}: {p:.4f}" for i, p in enumerate(prices)))

            elif cmd == "status":
                self.display_summary(success_outcome)

            elif cmd == "decide":
                rec = self.make_decision(success_outcome)
                all_prices = self.get_all_prices()
                print(f"  Recommended Decision: '{rec}' (success prob: {all_prices[rec][success_outcome]:.4f})")

            elif cmd in ("buy", "sell", "target", "budget"):
                if len(parts) < 4:
                    print(f"  Usage: {cmd} <action> <outcome> <value> {'[buy]' if cmd=='budget' else ''}")
                    continue

                # Handle actions with spaces by joining parts between the command and the last 2 (or 3) arguments
                is_budget_buy = (cmd == "budget" and len(parts) >= 5 and parts[-1].lower() == "buy")
                num_params = 3 if is_budget_buy else 2
                
                action = " ".join(parts[1:-num_params])
                try:
                    mkt = self.get_market(action)
                except KeyError as e:
                    print(f"  Error: {e}")
                    continue

                try:
                    outcome = int(parts[-num_params])
                    value = float(parts[-num_params + 1])

                    if cmd == "buy":
                        cost = mkt.buy_shares(outcome, value)
                        print(f"  [{action}] Bought {value:.4f} shares of Outcome {outcome}. Cost: {cost:.4f}")

                    elif cmd == "sell":
                        ret = mkt.sell_shares(outcome, value)
                        print(f"  [{action}] Sold {value:.4f} shares of Outcome {outcome}. Return: {ret:.4f}")

                    elif cmd == "target":
                        shares, cost = buy_to_target_probability(mkt, outcome, value)
                        if shares > 0:
                            print(f"  [{action}] Bought {shares:.4f} shares → Outcome {outcome} at p={value:.4f}. Cost: {cost:.4f}")
                        else:
                            print(f"  [{action}] Target probability already met or too low.")

                    elif cmd == "budget":
                        x = mkt.shares_from_budget(outcome, value)
                        print(f"  [{action}] With {value:.2f} you can buy {x:.4f} shares of Outcome {outcome}.")
                        if len(parts) >= 5 and parts[4].lower() == "buy":
                            cost = mkt.buy_shares(outcome, x)
                            print(f"  [{action}] Bought {x:.4f} shares. Cost: {cost:.4f}")

                    # Show updated prices for this market
                    prices = lmsr_prices(mkt.shares, mkt.b)
                    print(f"  [{action}] Updated prices: " + "  ".join(f"Outcome {i}: {p:.4f}" for i, p in enumerate(prices)))

                except (ValueError, IndexError) as e:
                    print(f"  Error: {e}")

            else:
                print(f"  Unknown command '{cmd}'. Commands: buy | sell | target | budget | prices | status | decide | quit")


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    actions = ["Invest in AI", "Invest in Biotech"]

    dm = DecisionMarket(actions=actions, b=100.0, n_outcomes=2, market_maker_fee=0.0)

    print("--- Decision Market Initialized ---")
    dm.display_summary()

    # Simulate agents trading based on their beliefs
    print("--- Agent Trading Simulation ---")

    # Agent 1: bullish on AI (believes 80% success if we invest in AI)
    print(">>> Agent 1: targets 80% success probability for 'Invest in AI'")
    buy_to_target_probability(dm.get_market("Invest in AI"), 0, 0.80)

    # Agent 2: bullish on Biotech (believes 70% success for Biotech)
    print(">>> Agent 2: targets 70% success probability for 'Invest in Biotech'")
    buy_to_target_probability(dm.get_market("Invest in Biotech"), 0, 0.70)

    # Agent 3: bear on AI (believes only 60% success for AI)
    print(">>> Agent 3: targets 60% success probability for 'Invest in AI'")
    buy_to_target_probability(dm.get_market("Invest in AI"), 1, 0.40)  # buys failure


    print()
    dm.display_summary()

    # Start the interactive session for manual trading
    dm.interactive_session()
