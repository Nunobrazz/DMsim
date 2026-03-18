import math
import numpy as np


def lmsr_total_value(q, b):
    """
    Calculates the total value (V) for a given state of shares (q).
        
    Parameters:
    q (list or array): The number of shares outstanding for each outcome.
    b (float): The liquidity parameter (the market maker's 'depth').
        
    Returns:
    float: The total cost value.
    """
    q = np.array(q)
        
    # Calculate the sum of exponents: sum(exp(q_i / b))
    sum_exponents = np.sum(np.exp(q / b))
        
    # Final formula: b * ln(sum_exponents)
    cost = b * np.log(sum_exponents)
        
    return cost

def lmsr_prices(q, b):
    """
    Calculates the prices (probabilities) for each outcome.
    
    Parameters:
    q (list or array): Number of shares outstanding for each outcome.
    b (float): Liquidity parameter (higher b = slower price movement).
    """
    q = np.array(q)
    
    # Calculate exponential values for each outcome
    exponents = np.exp(q / b)
    
    # Calculate price as the ratio of an outcome's exponent to the sum of all exponents
    prices = exponents / np.sum(exponents)
    
    return prices



class LMSR:
    """
    Represents a Prediction Market using the LMSR (Logarithmic Market Scoring Rule) algorithm.
    """

    def __init__(self, event, q, b, market_maker_fee):
        """
        Initializes the prediction market.
        
        Parameters:
        event (str): The name of the event.
        q (list or array): Initial number of shares outstanding for each outcome.
        b (float): The liquidity parameter (the market maker's 'depth').
        market_maker_fee (float): The fee charged by the market maker for transactions.
        """
        self.event = event
        self.b = float(b)
        self.shares = np.array(q, dtype=np.float64) # Number of shares for each outcome identified by id in array
        self.initial_shares = np.array(q, dtype=np.float64)
        self.market_maker_fee = market_maker_fee

    def get_event(self):
        """
        Retrieves the event associated with this prediction market.
        
        Returns:
        str: The name of the event.
        """
        return self.event

    def get_current_price(self, outcome):
        """
        Calculates the current price (probability) of a specific outcome.
        
        Parameters:
        outcome (int): The index identifying the outcome.
        
        Returns:
        float: The current price of the specified outcome.
        """
        return lmsr_prices(self.shares, self.b)[outcome]

    def get_market_value(self):
        """
        Retrieves the total value in the market.
        
        Returns:
        float: The total value in the market.
        """
        return lmsr_total_value(self.shares, self.b)

    def get_market_maker_fee(self):
        """
        Retrieves the market maker fee for transactions.
        
        Returns:
        float: The market maker fee.
        """
        return self.market_maker_fee
    
    def update_b(self):
        """
        Updates the liquidity parameter b.
        Currently not implemented; intended for use with LS-LMSR.
        """
        pass

    def tx_cost(self, outcome, n_shares):
        """
        Calculates the transaction cost from changing the number of shares.
        
        The cost is the difference between the new state and the old state.
        A positive cost indicates a buy, and a negative cost indicates a sell.
        
        Parameters:
        outcome (int): The index identifying the outcome.
        n_shares (float/int): The number of shares being transacted.
        
        Returns:
        float: The transaction cost.
        """
        new_shares = np.copy(self.shares)
        new_shares[outcome] += n_shares
        return lmsr_total_value(new_shares, self.b) - lmsr_total_value(self.shares, self.b)
    
    def buy_shares(self, outcome, shares):
        """
        Executes a buy transaction for a specified number of shares of an outcome.
        
        Parameters:
        outcome (int): The index identifying the outcome.
        shares (float/int): The number of shares to buy.
        
        Returns:
        float: The final cost of the buy transaction including the market maker fee.
        """
        price = self.get_current_price(outcome)
        cost = self.tx_cost(outcome, shares)
        self.shares[outcome] += shares
        self.update_b()
        return cost + self.get_market_maker_fee()

    def sell_shares(self, outcome, shares):
        """
        Executes a sell transaction for a specified number of shares of an outcome.
        
        Parameters:
        outcome (int): The index identifying the outcome.
        shares (float/int): The number of shares to sell.
        
        Returns:
        float: The final return of the sell transaction including the market maker fee.
        """
        cost = self.tx_cost(outcome, -shares)
        self.shares[outcome] -= shares
        self.update_b()
        return cost - self.get_market_maker_fee()

    def get_market_total_revenue(self):
        """
        Calculates the total revenue of the market maker after the event is resolved.
        
        Returns:
        float: The final total value minus the initial total value.
        """
        return lmsr_total_value(self.shares, self.b) - lmsr_total_value(self.initial_shares, self.b) 

    def get_market_total_PL(self, verified_outcome):
        """
        Calculates the P&L of the market maker after the event is resolved.
        
        Returns:
        float: The payout minus the total revenue.
        """
        winning_shares_sold = self.shares[verified_outcome] - self.initial_shares[verified_outcome]
        payout = winning_shares_sold * 1.0 
        
        PL = payout - self.get_market_total_revenue()
        return PL


    def shares_from_budget(self, outcome, amount):
        """
        Calculates exactly how many shares can be bought with a given amount.

        Derived from the LMSR cost function, the exact number of shares x that
        costs y euros at the current price p is:

            x = b * ln( (e^(y/b) - 1 + p) / p )

        Parameters:
        outcome (int): The index of the outcome to buy shares in.
        amount (float): The amount in euros to spend.

        Returns:
        float: The number of shares that can be bought with the given amount.
        """
        p = self.get_current_price(outcome)
        x = self.b * math.log((math.exp(amount / self.b) - 1 + p) / p)
        return x

    # TODO: This needs to be done with two different markets
    def optimal_report(self, profile):
        """
        Determines the optimal report for a given profile.
        
        Parameters:
        tuple (float, float, float, float): The profile of the reporter (theta_A, theta_B, p_A, p_B)
        
        Returns:
        int: The index identifying the outcome with the highest price.
        """
        pass

    def interactive_session(self):
        """
        Starts an interactive CLI session for participants to trade in the market.

        Commands:
            buy <outcome> <shares>       - Buy shares for an outcome.
            sell <outcome> <shares>      - Sell shares for an outcome.
            target <outcome> <prob>      - Buy shares to reach a target probability.
            prices                       - Show current prices for all outcomes.
            status                       - Show full market status.
            quit                         - Exit the session.
        """
        n_outcomes = len(self.shares)

        def _print_prices():
            prices = lmsr_prices(self.shares, self.b)
            print("  Current prices:")
            for i, p in enumerate(prices):
                print(f"    Outcome {i}: {p:.4f}")

        print(f"\n=== Interactive Market Session: '{self.event}' ===")
        print(f"  Outcomes: {n_outcomes} | Liquidity (b): {self.b} | Fee: {self.market_maker_fee}")
        _print_prices()
        print("  Commands: buy <outcome> <shares> | sell <outcome> <shares> | target <outcome> <prob> | prices | status | quit\n")

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
                _print_prices()

            elif cmd == "status":
                prices = lmsr_prices(self.shares, self.b)
                print(f"  Market: '{self.event}'")
                print(f"  Shares outstanding: {self.shares}")
                print(f"  Market Value (V): {lmsr_total_value(self.shares, self.b):.4f}")
                print(f"  Total Revenue: {self.get_market_total_revenue():.4f}")
                for i, p in enumerate(prices):
                    print(f"    Outcome {i}: price={p:.4f}, shares={self.shares[i]:.2f}")

            elif cmd == "buy":
                if len(parts) != 3:
                    print("  Usage: buy <outcome> <shares>")
                    continue
                try:
                    outcome = int(parts[1])
                    shares = float(parts[2])
                    cost = self.buy_shares(outcome, shares)
                    print(f"  Bought {shares:.4f} shares of Outcome {outcome}. Cost: {cost:.4f}")
                    _print_prices()
                except (ValueError, IndexError) as e:
                    print(f"  Error: {e}")

            elif cmd == "sell":
                if len(parts) != 3:
                    print("  Usage: sell <outcome> <shares>")
                    continue
                try:
                    outcome = int(parts[1])
                    shares = float(parts[2])
                    ret = self.sell_shares(outcome, shares)
                    print(f"  Sold {shares:.4f} shares of Outcome {outcome}. Return: {ret:.4f}")
                    _print_prices()
                except (ValueError, IndexError) as e:
                    print(f"  Error: {e}")

            elif cmd == "target":
                if len(parts) != 3:
                    print("  Usage: target <outcome> <probability>")
                    continue
                try:
                    outcome = int(parts[1])
                    prob = float(parts[2])
                    shares, cost = buy_to_target_probability(self, outcome, prob)
                    if shares > 0:
                        print(f"  Bought {shares:.4f} shares of Outcome {outcome} to reach p={prob:.4f}. Cost: {cost:.4f}")
                    _print_prices()
                except (ValueError, IndexError) as e:
                    print(f"  Error: {e}")

            elif cmd == "budget":
                if len(parts) not in (3, 4):
                    print("  Usage: budget <outcome> <amount> [buy]")
                    continue
                try:
                    outcome = int(parts[1])
                    amount = float(parts[2])
                    x = self.shares_from_budget(outcome, amount)
                    print(f"  With {amount:.2f} you can buy {x:.4f} shares of Outcome {outcome}.")
                    if len(parts) == 4 and parts[3].lower() == "buy":
                        cost = self.buy_shares(outcome, x)
                        print(f"  Bought {x:.4f} shares. Cost: {cost:.4f}")
                    _print_prices()
                except (ValueError, IndexError) as e:
                    print(f"  Error: {e}")

            else:
                print(f"  Unknown command '{cmd}'. Commands: buy | sell | target | budget | prices | status | quit")
        
        




    def shares_to_target_probability(self, outcome, target_probability):
        """
        Determines the number of shares to buy to reach a specific target probability,
        and executes the buy transaction.
        
        Parameters: 
        outcome (int): The index identifying the outcome.
        target_probability (float): The target probability/price (between 0 and 1).

        Returns:
        int: The shares the agnet should by to change the price to the target probability.
        """
        if target_probability <= 0 or target_probability >= 1:
            raise ValueError("Target probability must be strictly between 0 and 1.")

        current_price = self.get_current_price(outcome)
        if target_probability <= current_price:
            print("Target probability is not higher than the current probability.")
            return 0.0, 0.0

        # Calculate the sum of exponents for all OTHER outcomes (S_{-i})
        exponents = np.exp(self.shares / self.b)
        s_minus_i = np.sum(exponents) - exponents[outcome]
        
        # Calculate the exact delta_q needed
        delta_q = self.b * math.log((target_probability / (1.0 - target_probability)) * s_minus_i) - self.shares[outcome]
        
        return delta_q





def buy_to_target_probability(lmsr_market, outcome, target_probability):
    """
    Determines the number of shares to buy to reach a specific target probability,
    and executes the buy transaction.
    
    Parameters: 
    lmsr_market (LMSR): The LMSR market object.
    outcome (int): The index identifying the outcome.
    target_probability (float): The target probability/price (between 0 and 1).

    Returns:
    tuple(float, float): A tuple containing (shares_bought, total_cost).
    """
    if target_probability <= 0 or target_probability >= 1:
        raise ValueError("Target probability must be strictly between 0 and 1.")

    current_price = lmsr_market.get_current_price(outcome)
    if target_probability <= current_price:
        print("Target probability is not higher than the current probability.")
        return 0.0, 0.0

    # Calculate the sum of exponents for all OTHER outcomes (S_{-i})
    exponents = np.exp(lmsr_market.shares / lmsr_market.b)
    s_minus_i = np.sum(exponents) - exponents[outcome]
    
    # Calculate the exact delta_q needed
    delta_q = lmsr_market.b * math.log((target_probability / (1.0 - target_probability)) * s_minus_i) - lmsr_market.shares[outcome]
    
    cost = lmsr_market.buy_shares(outcome, delta_q)
    return delta_q, cost


# Example usage
if __name__ == "__main__":
    event_name = 'Invest in Bitcoin ?'
    b = 100
    q = np.array([0, 0]) # initial shares for each outcome
    market_maker_fee = 0

    market = LMSR(event_name, q, b, market_maker_fee)
    market.interactive_session()

    print(f"--- INIT ---")
    print(f"Event: {event_name}")
    print(f"Liquidity Parameter (b): {b}")
    print(f"Initial shares (q): {q}")
    print(f"Theoretical Max Market Maker Loss (b * ln(2)): {b * math.log(2):.2f}\n")
    print(f"Outcome 0 (No rain) price: {market.get_current_price(0):.4f}")
    print(f"Outcome 1 (Rain)    price: {market.get_current_price(1):.4f}\n")

    print(">>> Buying 10 shares of Outcome 0 (No rain)...")
    cost0 = market.buy_shares(0, 10)
    print(f"Cost: {cost0:.2f}")
    print(f"Outcome 0 (No rain) price: {market.get_current_price(0):.4f}")
    print(f"Outcome 1 (Rain)    price: {market.get_current_price(1):.4f}\n")

    print(">>> Buying 20 shares of Outcome 1 (Rain)...")
    cost1 = market.buy_shares(1, 20)
    print(f"Cost: {cost1:.2f}")
    print(f"Outcome 0 (No rain) price: {market.get_current_price(0):.4f}")
    print(f"Outcome 1 (Rain)    price: {market.get_current_price(1):.4f}\n")

    print(">>> Selling 5 shares of Outcome 0 (No rain)...")
    ret0 = market.sell_shares(0, 5)
    print(f"Return: {ret0:.2f}")
    print(f"Outcome 0 (No rain) price: {market.get_current_price(0):.4f}")
    print(f"Outcome 1 (Rain)    price: {market.get_current_price(1):.4f}\n")

    print(">>> Selling 10 shares of Outcome 1 (Rain)...")
    ret1 = market.sell_shares(1, 10)
    print(f"Return: {ret1:.2f}")
    print(f"Outcome 0 (No rain) price: {market.get_current_price(0):.4f}")
    print(f"Outcome 1 (Rain)    price: {market.get_current_price(1):.4f}\n")

    print(">>> Buying Outcome 0 to reach exactly 75% probability...")
    shares, target_cost = buy_to_target_probability(market, 0, 0.75)
    print(f"Shares bought: {shares:.4f} | Cost: {target_cost:.2f}")
    print(f"Outcome 0 (No rain) price: {market.get_current_price(0):.4f}")
    print(f"Outcome 1 (Rain)    price: {market.get_current_price(1):.4f}\n")

    print(f"Final Market Value: {lmsr_total_value(market.shares, market.b):.2f}")
    print(f"Market Maker Total Revenue: {market.get_market_total_revenue():.2f}") # subtracts b*ln(2) from the final market value
    print(f"Market Maker P&L (if Outcome 0 wins): {market.get_market_total_PL(0):.2f}")
    print(f"Market Maker P&L (if Outcome 1 wins): {market.get_market_total_PL(1):.2f}")
    