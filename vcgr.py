import numpy as np

class DecisionMechanism:
    """
    Represents a One-Shot Decision-Making Game based on a report profile.
    """

    def __init__(self, reports, budget_constraint):
        """
        Initializes the decision mechanism.
        
        Parameters:
        reports (list or array): The report profile m = (m_1, ..., m_n) where m_i is a float.
        budget_constraint (float): The budget constraint c for the mechanism.
        """
        self.m = np.array(reports, dtype=np.float64)
        self.n = len(self.m)
        self.c = float(budget_constraint)


    def report(self, agent_index, report_value):
        """
        Updates the report profile with a new report from a specific agent.
        
        Parameters:
        agent_index (int): The index of the agent (0 to n-1).
        report_value (float): The value reported by the agent.
        """
        if 0 <= agent_index < self.n:
            self.m[agent_index] = float(report_value)
        else:
            raise IndexError("Agent index out of bounds.")

    def get_allocation(self):
        """
        Determines the allocation (action) based on the sum of the reports.
        
        Returns:
        int: '1' if the sum of reports is >= 0, otherwise '0'.
        """
        return np.sum(self.m) >= 0

    def calculate_t(self):
        """
        Calculates the payment/transfer rule t_i for each agent based on the pivot logic.
        
        Returns:
        numpy.ndarray: An array of t_i values for each agent.
        """
        t = np.zeros(self.n, dtype=np.float64)
        M = np.sum(self.m)
        
        for i in range(self.n):
            m_i = self.m[i]
            
            # S_minus_i is the sum of reports from all agents EXCEPT agent i
            S_minus_i = M - m_i
            
            if -m_i > S_minus_i > 0:
                t[i] = -S_minus_i
            elif -m_i < S_minus_i < 0:
                t[i] = S_minus_i
            else:
                t[i] = 0.0
                
        return t

    def resolve_game(self, delta):
        """
        Observes the realized outcome metric delta and calculates final rewards and payoffs.
        
        Parameters:
        delta (float): The realized outcome metric (\\Delta).
        
        Returns:
        tuple: (allocation, t, r, pi) containing the chosen action, transfers, rewards, and final payoffs.
        """
        allocation = self.get_allocation()
        t = self.calculate_t()
        
        r = np.zeros(self.n, dtype=np.float64)
        pi = np.zeros(self.n, dtype=np.float64)
        
        for i in range(self.n):
            if delta > 0:
                r[i] = -t[i] + (self.c / self.n)
            else:
                r[i] = t[i] - (self.c / self.n)
            
            # Final payoff is the sum of the transfer and the reward
            pi[i] = t[i] + r[i]
            
        return allocation, t, r, pi

    def optimal_report(self, profile):
        """
        Calculates the optimal report for an agent based on their type.
        
        Parameters:
        theta_A (float): Monetary value/payoff for outcome A (\theta_i^A)
        theta_B (float): Monetary value/payoff for outcome B (\theta_i^B)
        p_A (float): Belief that outcome A will succeed (p_i^A)
        p_B (float):  Belief that outcome B will succee (p_i^B)
        c (float): Budget constraint (c)
        n (int): Number of agents (n)
        
        Returns:
        float: The optimal report
        """
        theta_A, theta_B, p_A, p_B = profile
        if theta_A > theta_B:
            if p_A == 1.0:
                raise ValueError("p_A cannot be 1.0 as it causes division by zero.")
                
            idiosyncratic = (theta_A - theta_B) / (2 * (1 - p_A))
            information = (self.c / self.c) * ((p_A - p_B) / (1 - p_A))
            return idiosyncratic + information

        # Handle the case where theta_B > theta_A
        elif theta_B > theta_A:
            if p_B == 1.0:
                raise ValueError("p_B cannot be 1.0 as it causes division by zero.")
                
            idiosyncratic = (theta_A - theta_B) / (2 * (1 - p_B))
            information = (self.c / self.c) * ((p_A - p_B) / (1 - p_B))
            return idiosyncratic + information
            
        # Handle the case where theta_A == theta_B (indifference)
        else:
            # If the agent is indifferent, the first term evaluates to 0.
            return 0.0



    def display_summary(self, delta):
        """
        Prints a summary of the game resolution including reports, transfers, rewards, and payoffs.
        
        Parameters:
        delta (float): The realized outcome metric (\\Delta).
        """
        allocation, t, r, pi = self.resolve_game(delta)
        M = np.sum(self.m)
        
        print(f"=== VCGR GAME RESOLUTION (Δ = {delta}) ===")
        print(f"Final Report Profile (m): {self.m}")
        print(f"Final Sum of reports (M): {M:.2f}")
        print(f"Final Allocation: {allocation}\n")
        
        for i in range(self.n):
            print(f"Agent {i+1} | Report: {self.m[i]:>6.1f} | Transfer t_{i}: {t[i]:>6.2f} | Reward r_{i}: {r[i]:>6.2f} | Payoff (pi_{i}): {pi[i]:>6.2f}")
        print()
    



# Example usage
if __name__ == "__main__":
    # Define a scenario with 3 agents starting with no reports (0.0)
    reports = [0.0, 0.0, 0.0] 
    budget = 5.0
    
    game = DecisionMechanism(reports, budget)

    print(f"--- INIT ---")
    print(f"Number of agents (n): {game.n}")
    print(f"Budget constraint (c): {game.c}")
    print(f"Initial Report Profile (m): {game.m}\n")
    
    print(">>> Agent 1 submits report: 10.0")
    game.report(0, 10.0)
    print(f"Current Report Profile (m): {game.m}")
    print(f"Current Tentative Allocation (a): {game.get_allocation()}\n")

    print(">>> Agent 2 submits report: -4.0")
    game.report(1, -4.0)
    print(f"Current Report Profile (m): {game.m}")
    print(f"Current Tentative Allocation (a): {game.get_allocation()}\n")

    print(">>> Agent 3 submits report: -8.0")
    game.report(2, -8.0)
    print(f"Current Report Profile (m): {game.m}")
    print(f"Current Tentative Allocation (a): {game.get_allocation()}\n")

    # Resolve game with final profile
    delta_positive = 5.0
    game.display_summary(delta_positive)