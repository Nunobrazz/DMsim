import numpy as np
import os
from profile_generator import generate_profiles, DAO_SCENARIO, build_matrix, Colors, display_profiles
from organization import Organization
from vcgr import DecisionMechanism

def run_simulation():
    print(f"\n{Colors.BOLD}{Colors.CYAN}🚀 STARTING DAO GOVERNANCE SIMULATION{Colors.END}")
    print(f"{'─'*40}")

    # 1. Generate Profiles
    # Note: Ensure GOOGLE_API_KEY or GROQ_API_KEY is in environment
    try:
        profiles = generate_profiles(DAO_SCENARIO, n_agents=5)
        display_profiles(profiles)
    except Exception as e:
        print(f"{Colors.RED}Failed to generate profiles: {e}{Colors.END}")
        return

    # 2. Initialize Organization
    matrix = build_matrix(profiles)
    org = Organization(matrix)
    
    # 3. Setup VCG Mechanism
    # We use a budget constraint (c) to control the 'signal' strength of reports
    budget = 100.0  
    
    # We'll simulate the "optimal report" strategy
    # Agents will report m_i based on their private theta and p
    reports = np.zeros(org.n)
    vcg = DecisionMechanism(reports, budget)

    print(f"\n{Colors.BOLD}{Colors.HEADER}⚖️  VCGR MECHANISM: STRATEGIC REPORTING {Colors.END}")
    print(f"{'─'*40}")

    for i in range(org.n):
        profile = org.get_agent_profile(i)

        # Optimal report based on the VCGR logic (Truthful representation of expected value + strategic signal)
        m_i = vcg.optimal_report(profile)
        vcg.report(i, m_i)
        
        status = f"{Colors.GREEN if m_i > 0 else Colors.RED}{'SUPPORT A' if m_i > 0 else 'SUPPORT B'}{Colors.END}"
        print(f"Agent {i} ({profiles[i].name[:20]:<20}) reported: {m_i:>8.2f} | {status}")

    # 4. Resolve the decision
    # We need a delta (realized outcome) to resolve transfers, but for the decision itself,
    # we just look at the sum of reports.
    allocation = vcg.get_allocation()
    sum_reports = np.sum(vcg.m)
    
    print(f"\n{Colors.BOLD}{Colors.YELLOW}┌{'─'*48}┐{Colors.END}")
    decision_str = f"ACTION {'A' if allocation else 'B'} APPROVED"
    print(f"{Colors.BOLD}{Colors.YELLOW}│ {decision_str:^46} │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}└{'─'*48}┘{Colors.END}")
    print(f"Total Combined weight (M): {sum_reports:.2f}")

    # 5. Theoretical resolution (assuming delta = 1 for Action A or -1 for Action B)
    # This shows the transfers and payoffs if the chosen action turns out "good"
    delta = np.random.choice([1.0, -1.0])
    print(f"\n{Colors.CYAN}Outcome simulated with Δ = {delta} (Action Success){Colors.END}")
    vcg.display_summary(delta)

if __name__ == "__main__":
    run_simulation()
