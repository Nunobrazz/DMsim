import numpy as np

class Organization:
    """
    Represents an Organization consisting of n agents.
    Each agent i is defined by the profile: (theta_A_i, theta_B_i, p_A_i, p_B_i)
    where 'theta' represents idiosyncratic preferences and 'p' represents probabilities.
    """

    def __init__(self, agent_profiles):
        """
        Initializes the organization with the agents' profiles.
        
        Parameters:
        agent_profiles (list of lists or numpy.ndarray): An n x 4 matrix where each row is an agent
                                                         and columns are [theta_A, theta_B, p_A, p_B].
        """
        self.profiles = np.array(agent_profiles, dtype=np.float64)
        self.n = len(self.profiles)
        
        # Extract individual parameter arrays for easier vectorized calculations
        self.theta_A = self.profiles[:, 0]
        self.theta_B = self.profiles[:, 1]
        self.p_A = self.profiles[:, 2]
        self.p_B = self.profiles[:, 3]

    def get_agent_profile(self, agent_index):
        """
        Retrieves the profile of a specific agent.
        
        Parameters:
        agent_index (int): The index of the agent (0 to n-1).
        
        Returns:
        numpy.ndarray: The array [theta_A, theta_B, p_A, p_B] for the specified agent.
        """
        return self.profiles[agent_index]

    def expected_utilities_A(self, p_A_override=None):
        probs = self.p_A if p_A_override is None else p_A_override
        return self.theta_A * probs

    def expected_utilities_B(self, p_B_override=None):
        probs = self.p_B if p_B_override is None else p_B_override
        return self.theta_B * probs

    def utilitarian_welfare(self, p_A_override=None, p_B_override=None):
        return np.sum(self.expected_utilities_A(p_A_override)), np.sum(self.expected_utilities_B(p_B_override))

    def utilitarian_decision(self, p_A_override=None, p_B_override=None):
        w_A, w_B = self.utilitarian_welfare(p_A_override, p_B_override)
        return 'A' if w_A >= w_B else 'B'

    def get_vcg_reports(self, p_A_override=None, p_B_override=None):
        return self.expected_utilities_A(p_A_override) - self.expected_utilities_B(p_B_override)

    def display_summary(self):
        """
        Prints a summary of the organization's agents and their expected utilities.
        """
        
        print(f"{'Agent':<7} | {'θ_A':<6} | {'θ_B':<6} | {'p_A':<5} | {'p_B':<5}")
        print("-" * 65)
        for i in range(self.n):
            print(f"{i:<7} | {self.theta_A[i]:<6.1f} | {self.theta_B[i]:<6.1f} | {self.p_A[i]:<5.2f} | {self.p_B[i]:<5.2f}")


# Example usage
if __name__ == "__main__":
    from vcgr import DecisionMechanism
    from mbsr import LMSR, buy_to_target_probability

    # Define profiles for 4 agents: [theta_A, theta_B, p_A, p_B]
    # Note: probabilities do not strictly need to sum to 1 depending on whether 
    # they are conditional probabilities or mutually exclusive independent events.
    agent_data = [
        [100.0,  50.0, 0.6, 0.3],  # Agent 0: Strongly prefers A, believes A is likely
        [-20.0, 150.0, 0.4, 0.5],  # Agent 1: Loses on A, wins big on B, believes B is more likely
        [ 10.0,  10.0, 0.3, 0.5],  # Agent 2: Indifferent between outcomes, uncertain
        [ 80.0, -10.0, 0.7, 0.4]   # Agent 3: Prefers A, highly confident in A
    ]
    
    org = Organization(agent_data)
    
    print("\n=== Organization Initialization ===")
    print(f"Total Agents (n): {org.n}\n")
    org.display_summary()
    print('\n')


    reports_private = org.get_vcg_reports()
    budget = 100.0
    delta = 1
    vcgr = DecisionMechanism(reports_private, budget)
    agent = 0
    for profile in org.profiles:
        m_i = vcgr.optimal_report(profile)
        vcgr.report(agent, m_i)
        agent += 1 
    vcgr.resolve_game(delta)
    vcgr.display_summary(delta)


    print("--- 3. LMSR as Decision Market ---")
    # Initialize a market for A (outcome 0) vs B (outcome 1)
    market = LMSR("Organizational Decision", [0, 0], 100.0, 0.0)
    print(f"Initial Market Prices: A={market.get_current_price(0):.2f}, B={market.get_current_price(1):.2f}")
    
    for i in range(org.n):
        # Each agent trades to push the market towards their private belief for A
        target_A = org.p_A[i]
        curr_A = market.get_current_price(0)
        if target_A > curr_A:
            buy_to_target_probability(market, 0, target_A)
            print(f"Agent {i} bought A to target p_A = {target_A:.2f}. New Market Price A = {market.get_current_price(0):.4f}")
        elif target_A < curr_A:
            # To lower p_A, buy B to target 1 - target_A
            buy_to_target_probability(market, 1, 1.0 - target_A)
            print(f"Agent {i} bought B to target p_A = {target_A:.2f}. New Market Price A = {market.get_current_price(0):.4f}")
        else:
            print(f"Agent {i} matches market. New Market Price A = {market.get_current_price(0):.4f}")
    
    consensus_p_A = market.get_current_price(0)
    consensus_p_B = market.get_current_price(1)
    print(f"\n>>> Final Consensus Probabilities: p_A={consensus_p_A:.4f}, p_B={consensus_p_B:.4f}\n")

    print("--- 4. VCG Mechanism (Consensus Beliefs) ---")
    reports_consensus = org.get_vcg_reports(p_A_override=consensus_p_A, p_B_override=consensus_p_B)
    vcg_consensus = DecisionMechanism(reports_consensus, budget)
    print(f"Consensus Reports: {vcg_consensus.m}")
    print(f">>> VCG Mechanism Allocation (Consensus Beliefs): {vcg_consensus.get_allocation()}")
    