"""
Cloud Server Cost Optimization for a Web Application
Linear Programming Model Implementation
"""

# Import statements
import pulp
import pandas as pd
import matplotlib.pyplot as plt

# Part A: Data Definitions

# Define sets
periods = ["OffPeak", "Normal", "Peak"]
instance_types = ["Small", "Medium", "Large"]

# Define parameters as dictionaries
# Cost per hour for each instance type (in dollars)
cost = {
    "Small": 0.06,
    "Medium": 0.10,
    "Large": 0.18
}

# CPU capacity for each instance type (CPU units)
cpu = {
    "Small": 1,
    "Medium": 2,
    "Large": 4
}

# RAM capacity for each instance type (gigabytes)
ram = {
    "Small": 2,
    "Medium": 4,
    "Large": 8
}

# CPU demand for each period (CPU units)
cpu_demand = {
    "OffPeak": 6,
    "Normal": 10,
    "Peak": 18
}

# RAM demand for each period (gigabytes)
ram_demand = {
    "OffPeak": 12,
    "Normal": 20,
    "Peak": 36
}

# Part D: Scenario Function (Reusable Solver)


def solve_scenario(cpu_demand_modifications=None, ram_demand_modifications=None,
                   cost_modifications=None):
    """
    Solve the cloud optimization problem with modified parameters.

    Parameters:
    -----------
    cpu_demand_modifications : dict, optional
        Dictionary with period keys and modified CPU demand values
        Example: {"Peak": 9.6} to modify Peak CPU demand
    ram_demand_modifications : dict, optional
        Dictionary with period keys and modified RAM demand values
        Example: {"Peak": 19.2} to modify Peak RAM demand
    cost_modifications : dict, optional
        Dictionary with instance type keys and modified cost values
        Example: {"Large": 0.46} to modify Large instance cost

    Returns:
    --------
    dict : Dictionary containing:
        - 'total_cost': Total hourly cost
        - 'results_df': DataFrame with allocation results
        - 'status': Solver status
    """
    # Start with base parameters
    scenario_cpu_demand = cpu_demand.copy()
    scenario_ram_demand = ram_demand.copy()
    scenario_cost = cost.copy()

    # Apply modifications if provided
    if cpu_demand_modifications:
        scenario_cpu_demand.update(cpu_demand_modifications)

    if ram_demand_modifications:
        scenario_ram_demand.update(ram_demand_modifications)

    if cost_modifications:
        scenario_cost.update(cost_modifications)

    # Create new LP problem
    scenario_prob = pulp.LpProblem(
        "Cloud_Server_Cost_Optimization_Scenario", pulp.LpMinimize)

    # Decision variables
    scenario_x = {}
    for period in periods:
        scenario_x[period] = {}
        for instance_type in instance_types:
            var_name = f"x_{period}_{instance_type}_scenario"
            scenario_x[period][instance_type] = pulp.LpVariable(
                var_name,
                lowBound=0,
                cat='Integer'
            )

    # Objective function
    scenario_prob += pulp.lpSum([
        scenario_cost[instance_type] * scenario_x[period][instance_type]
        for period in periods
        for instance_type in instance_types
    ]), "Total_Cost"

    # Constraints
    for period in periods:
        # CPU constraint
        scenario_prob += pulp.lpSum([
            cpu[instance_type] * scenario_x[period][instance_type]
            for instance_type in instance_types
        ]) >= scenario_cpu_demand[period], f"CPU_Constraint_{period}"

        # RAM constraint
        scenario_prob += pulp.lpSum([
            ram[instance_type] * scenario_x[period][instance_type]
            for instance_type in instance_types
        ]) >= scenario_ram_demand[period], f"RAM_Constraint_{period}"

        # Redundancy constraint
        scenario_prob += pulp.lpSum([
            scenario_x[period][instance_type]
            for instance_type in instance_types
        ]) >= 2, f"Redundancy_Constraint_{period}"

    # Solve
    scenario_status = scenario_prob.solve()

    # Extract results if optimal
    if scenario_status == pulp.LpStatusOptimal:
        results_data = []
        for period in periods:
            small_count = int(pulp.value(scenario_x[period]["Small"]))
            medium_count = int(pulp.value(scenario_x[period]["Medium"]))
            large_count = int(pulp.value(scenario_x[period]["Large"]))

            period_cost = (
                scenario_cost["Small"] * small_count +
                scenario_cost["Medium"] * medium_count +
                scenario_cost["Large"] * large_count
            )

            results_data.append({
                "Period": period,
                "Small": small_count,
                "Medium": medium_count,
                "Large": large_count,
                "Period Cost ($)": period_cost
            })

        results_df = pd.DataFrame(results_data)
        total_cost = results_df["Period Cost ($)"].sum()

        return {
            'total_cost': total_cost,
            'results_df': results_df,
            'status': pulp.LpStatus[scenario_status]
        }
    else:
        return {
            'total_cost': None,
            'results_df': None,
            'status': pulp.LpStatus[scenario_status]
        }


# Main execution block
if __name__ == "__main__":
    # Part A: Display data tables
    print("=" * 60)
    print("Instance Parameters")
    print("=" * 60)
    instance_df = pd.DataFrame({
        "Instance Type": instance_types,
        "Cost per Hour ($)": [cost[t] for t in instance_types],
        "CPU Capacity": [cpu[t] for t in instance_types],
        "RAM Capacity (GB)": [ram[t] for t in instance_types]
    })
    print(instance_df.to_string(index=False))
    print()

    print("=" * 60)
    print("Resource Demand per Period")
    print("=" * 60)
    demand_df = pd.DataFrame({
        "Period": periods,
        "CPU Demand": [cpu_demand[p] for p in periods],
        "RAM Demand (GB)": [ram_demand[p] for p in periods]
    })
    print(demand_df.to_string(index=False))
    print()

    # Part B: Build and solve base case
    print("=" * 60)
    print("Solving the Linear Programming Problem...")
    print("=" * 60)

    # Create LP minimization problem
    prob = pulp.LpProblem("Cloud_Server_Cost_Optimization", pulp.LpMinimize)

    # Decision variables
    # x[t][k]: number of instances of type k running in period t
    # Variables are integers and nonnegative
    x = {}
    for period in periods:
        x[period] = {}
        for instance_type in instance_types:
            var_name = f"x_{period}_{instance_type}"
            x[period][instance_type] = pulp.LpVariable(
                var_name,
                lowBound=0,
                cat='Integer'
            )

    # Objective function
    # Minimize total cost: sum over all periods and instance types of cost[type] * x[period][type]
    prob += pulp.lpSum([
        cost[instance_type] * x[period][instance_type]
        for period in periods
        for instance_type in instance_types
    ]), "Total_Cost"

    # Constraints
    # CPU constraint for each period
    for period in periods:
        prob += pulp.lpSum([
            cpu[instance_type] * x[period][instance_type]
            for instance_type in instance_types
        ]) >= cpu_demand[period], f"CPU_Constraint_{period}"

    # RAM constraint for each period
    for period in periods:
        prob += pulp.lpSum([
            ram[instance_type] * x[period][instance_type]
            for instance_type in instance_types
        ]) >= ram_demand[period], f"RAM_Constraint_{period}"

    # Redundancy constraint for each period
    for period in periods:
        prob += pulp.lpSum([
            x[period][instance_type]
            for instance_type in instance_types
        ]) >= 2, f"Redundancy_Constraint_{period}"

    # Solve the problem
    status = prob.solve()

    # Capture solver status
    print(f"Solver Status: {pulp.LpStatus[status]}")
    if status == pulp.LpStatusOptimal:
        print("Optimal solution found!")
        print(f"Total Cost: ${pulp.value(prob.objective):.2f}")
    else:
        print("Warning: Optimal solution not found!")
    print()

    # Part C: Print results
    if status == pulp.LpStatusOptimal:
        # Extract variable values and compute period costs
        results_data = []

        for period in periods:
            # Extract integer values for each instance type
            small_count = int(pulp.value(x[period]["Small"]))
            medium_count = int(pulp.value(x[period]["Medium"]))
            large_count = int(pulp.value(x[period]["Large"]))

            # Compute cost for this period
            period_cost = (
                cost["Small"] * small_count +
                cost["Medium"] * medium_count +
                cost["Large"] * large_count
            )

            # Store results
            results_data.append({
                "Period": period,
                "Small": small_count,
                "Medium": medium_count,
                "Large": large_count,
                "Period Cost ($)": period_cost
            })

        # Build DataFrame
        results_df = pd.DataFrame(results_data)

        # Compute total cost
        total_cost = results_df["Period Cost ($)"].sum()

        # Add total row
        total_row = pd.DataFrame({
            "Period": ["Total hourly cost:"],
            "Small": [""],
            "Medium": [""],
            "Large": [""],
            "Period Cost ($)": [total_cost]
        })
        results_df = pd.concat([results_df, total_row], ignore_index=True)

        # Print results table
        print("=" * 60)
        print("Optimal Instance Allocation and Costs")
        print("=" * 60)
        print(results_df.to_string(index=False))
        print()

        # Print total cost separately
        print(f"Total Hourly Cost: ${total_cost:.2f}")
        print()

        # Store the results for later use
        base_case_results = results_df.copy()
        base_case_total_cost = total_cost
    else:
        print("Cannot extract results - optimal solution not found.")
        base_case_results = None
        base_case_total_cost = None

    # Part E: Generate and save plot
    if base_case_results is not None:
        # Prepare data for plotting (exclude the total row)
        plot_df = base_case_results[base_case_results['Period']
                                    != 'Total hourly cost:'].copy()

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up bar positions
        x_pos = range(len(plot_df))
        width = 0.25  # Width of bars

        # Create bars for each instance type
        small_bars = ax.bar([i - width for i in x_pos], plot_df['Small'], width,
                            label='Small', color='#3498db')
        medium_bars = ax.bar(x_pos, plot_df['Medium'], width,
                             label='Medium', color='#2ecc71')
        large_bars = ax.bar([i + width for i in x_pos], plot_df['Large'], width,
                            label='Large', color='#e74c3c')

        # Customize the plot
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Number of Instances', fontsize=12)
        ax.set_title('Optimal Instance Allocation by Period',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_df['Period'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bars in [small_bars, medium_bars, large_bars]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=9)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the figure
        plt.savefig('instance_allocation.png', dpi=300, bbox_inches='tight')
        print("=" * 60)
        print("Plot saved as 'instance_allocation.png'")
        print("=" * 60)
        print()

        # Optionally display the plot (comment out if running in headless mode)
        # plt.show()
        plt.close()
    else:
        print("Cannot create plot - base case results not available.")
        print()

    # Part D: Run scenarios and print their costs
    # Run Scenario A: Peak demand increases by 20%
    print("=" * 60)
    print("Scenario A: Peak Demand +20%")
    print("=" * 60)

    # Calculate modified Peak demand (20% increase)
    peak_cpu_demand_modified = cpu_demand["Peak"] * 1.2
    peak_ram_demand_modified = ram_demand["Peak"] * 1.2

    scenario_a = solve_scenario(
        cpu_demand_modifications={"Peak": peak_cpu_demand_modified},
        ram_demand_modifications={"Peak": peak_ram_demand_modified}
    )

    if scenario_a['total_cost'] is not None:
        print(f"Total Cost: ${scenario_a['total_cost']:.2f}")
        print("\nAllocation:")
        print(scenario_a['results_df'].to_string(index=False))

        # Generate allocation description
        peak_allocation = scenario_a['results_df'][scenario_a['results_df']
                                                   ['Period'] == 'Peak'].iloc[0]
        scenario_a_description = (
            f"Peak: {int(peak_allocation['Small'])} Small, "
            f"{int(peak_allocation['Medium'])} Medium, "
            f"{int(peak_allocation['Large'])} Large"
        )
    else:
        print(f"Scenario A failed: {scenario_a['status']}")
        scenario_a_description = "Solution not found"
    print()

    # Run Scenario B: Large instance cost increases by 15%
    print("=" * 60)
    print("Scenario B: Large Instance Cost +15%")
    print("=" * 60)

    # Calculate modified Large cost (15% increase)
    large_cost_modified = cost["Large"] * 1.15

    scenario_b = solve_scenario(
        cost_modifications={"Large": large_cost_modified}
    )

    if scenario_b['total_cost'] is not None:
        print(f"Total Cost: ${scenario_b['total_cost']:.2f}")
        print("\nAllocation:")
        print(scenario_b['results_df'].to_string(index=False))

        # Generate allocation description
        # Check which instance types are used across periods
        total_small = scenario_b['results_df']['Small'].sum()
        total_medium = scenario_b['results_df']['Medium'].sum()
        total_large = scenario_b['results_df']['Large'].sum()

        if total_large > 0 and total_medium > 0:
            scenario_b_description = "Shift from Large to Medium instances"
        elif total_large == 0:
            scenario_b_description = "No Large instances used, more Medium/Small"
        else:
            scenario_b_description = "Large instances still dominate"
    else:
        print(f"Scenario B failed: {scenario_b['status']}")
        scenario_b_description = "Solution not found"
    print()

    # Store scenario results for scenario analysis table
    # Generate base case description from results
    if base_case_results is not None:
        base_plot_df = base_case_results[base_case_results['Period']
                                         != 'Total hourly cost:'].copy()
        total_base_small = base_plot_df['Small'].sum()
        total_base_medium = base_plot_df['Medium'].sum()
        total_base_large = base_plot_df['Large'].sum()

        if total_base_large > 0:
            if total_base_medium > 0:
                base_description = "Large dominates, Medium appears"
            else:
                base_description = "Large dominates"
        elif total_base_medium > 0:
            base_description = "Medium dominates"
        else:
            base_description = "Small instances used"
    else:
        base_description = "Large dominates, Medium appears, etc."

    scenario_results = {
        "Base": {
            "total_cost": base_case_total_cost if base_case_results is not None else None,
            "description": base_description
        },
        "Peak +20%": {
            "total_cost": scenario_a['total_cost'],
            "description": scenario_a_description
        },
        "Large +15%": {
            "total_cost": scenario_b['total_cost'],
            "description": scenario_b_description
        }
    }

    # Print scenario comparison table
    print("=" * 60)
    print("Scenario Comparison")
    print("=" * 60)
    comparison_data = []
    for scenario_name, scenario_info in scenario_results.items():
        if scenario_info['total_cost'] is not None:
            comparison_data.append({
                "Scenario": scenario_name,
                "Total Cost ($)": f"${scenario_info['total_cost']:.2f}",
                "High-Level Allocation Notes": scenario_info['description']
            })
        else:
            comparison_data.append({
                "Scenario": scenario_name,
                "Total Cost ($)": "N/A",
                "High-Level Allocation Notes": scenario_info['description']
            })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
