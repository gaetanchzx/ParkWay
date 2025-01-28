import numpy as np
from scipy.optimize import linprog

# Example of fictitious data
parkings = ["P1", "P2", "P3"]  # Parking names
capacities = [50, 30, 20]  # Parking capacities
costs = [10, 7, 5]  # Parking usage costs
proximities = [0.9, 0.7, 0.5]  # Proximity to a target location (score from 0 to 1)
demands = [20, 15, 10]  # User demands for each parking
traffic_levels = [0.8, 0.5, 0.3]  # Traffic levels around each parking (score from 0 to 1)
user_preferences = [0.7, 0.9, 0.6]  # User preferences for each parking (score from 0 to 1)

# Objective: Minimize costs (-favor proximity + penalize traffic)
cost_weights = np.array(costs)
proximity_weights = np.array([1 - p for p in proximities])
traffic_penalty = np.array(traffic_levels)
preference_bonus = np.array([1 - p for p in user_preferences])

# Final weighting: cost + proximity + traffic + preferences
objective_weights = cost_weights + proximity_weights + traffic_penalty - preference_bonus

# Constraints
A_eq = [np.ones(len(parkings))]  # Sum of allocations = total number of cars to place
b_eq = [sum(demands)]

A_ub = np.eye(len(parkings))  # Allocations < capacity of each parking
b_ub = capacities

# Additional constraint: a parking lot cannot be underutilized below 50% if used
min_utilization = 0.5
A_ub_extra = -np.eye(len(parkings))
b_ub_extra = [-min_utilization * capacity if demand > 0 else 0 for capacity, demand in zip(capacities, demands)]
A_ub = np.vstack([A_ub, A_ub_extra])
b_ub = np.hstack([b_ub, b_ub_extra])

# Variable bounds (values cannot exceed capacities)
bounds = [(0, capacity) for capacity in capacities]

# Solve with linprog
try:
    result = linprog(
        c=objective_weights,  # Function to minimize
        A_eq=A_eq, b_eq=b_eq,  # Equality constraints
        A_ub=A_ub, b_ub=b_ub,  # Inequality constraints
        bounds=bounds,  # Variable bounds
        method="highs"
    )

    # Result
    if result.success:
        allocation = result.x
        print("Optimal allocation of cars:")
        for i, parking in enumerate(parkings):
            print(f"{parking}: {int(allocation[i])} cars")
    else:
        print("Optimization problem not solved. Using an alternative allocation...")
        # Simple alternative allocation based on available capacities
        total_demand = sum(demands)
        allocation = [min(demand, capacity) for demand, capacity in zip(demands, capacities)]
        remaining_demand = total_demand - sum(allocation)

        if remaining_demand > 0:
            for i in range(len(allocation)):
                available_capacity = capacities[i] - allocation[i]
                additional_allocation = min(available_capacity, remaining_demand)
                allocation[i] += additional_allocation
                remaining_demand -= additional_allocation
                if remaining_demand <= 0:
                    break

        print("Alternative allocation of cars:")
        for i, parking in enumerate(parkings):
            print(f"{parking}: {allocation[i]} cars")

except Exception as e:
    print("An error occurred during problem solving. Using an alternative allocation...")
    # Simple alternative allocation in case of an error
    total_demand = sum(demands)
    allocation = [min(demand, capacity) for demand, capacity in zip(demands, capacities)]
    remaining_demand = total_demand - sum(allocation)

    if remaining_demand > 0:
        for i in range(len(allocation)):
            available_capacity = capacities[i] - allocation[i]
            additional_allocation = min(available_capacity, remaining_demand)
            allocation[i] += additional_allocation
            remaining_demand -= additional_allocation
            if remaining_demand <= 0:
                break

    print("Alternative allocation of cars:")
    for i, parking in enumerate(parkings):
        print(f"{parking}: {allocation[i]} cars")

