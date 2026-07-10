# -*- coding: utf-8 -*-
"""
Created on Sat May 30 05:36:20 2026

@author: emmallen
"""

from xml.parsers.expat import model

import networkx as nx
import pyomo.environ as pyo
import time


def build_instance_data(G, s, t, problem_type="shortest_path"):
    
    '''Convert graph into sets and parameters for the MIP model.

   Returns:
   - nodes: list of nodes
   - arcs: list of directed edges
   - cost: arc length dictionary
   - capacity: arc capacity dictionary
   - penalty: interdiction penalty (currently uniform with value 1)
   - supply: flow balance (+1 at source, -1 at sink)'''
   
   # list of nodes
    nodes = list(G.nodes())
    # list of edges
    arcs = list(G.edges())

    # interdiction penalty = currently set to 1
    penalty = {(u, v): G[u][v]["penalty"] for (u, v) in arcs}

    # flow balance supply values
    supply = {i: 0 for i in nodes}
    supply[s] = 1
    supply[t] = -1

    cost = None
    capacity = None

    if problem_type in ["shortest_path", "min_cost_flow"]:
        cost = {(u, v): G[u][v]["dist"] for (u, v) in arcs}

    if problem_type in ["max_flow", "min_cost_flow"]:
        capacity = {(u, v): G[u][v]["capacity"] for (u, v) in arcs}

    if problem_type not in ["shortest_path", "max_flow", "min_cost_flow"]:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    return nodes, arcs, cost, capacity, penalty, supply

def build_training_sample(G,s,t,density,attack_limit,arcs,penalty,model,objective_name,objective_value,
                          mip_solve_time,status,termination,cost=None,capacity=None):

    """Construct the shared JSON training-sample structure."""
    
    # created sorted edge list
    edge_list = sorted(arcs)

    # cosntruct training sample
    sample = {
        "n_nodes": G.number_of_nodes(),
        "n_arcs": G.number_of_edges(),
        "density": density,
        "source": s,
        "sink": t,
        "attack_limit": attack_limit,

        # Arc information
        "u": [u for u, v in edge_list],
        "v": [v for u, v in edge_list],
        "penalty": [penalty[u, v]for u, v in edge_list],

        # Optimal interdiction decisions
        "attack": [int(round(pyo.value(model.Y[u, v]))) for u, v in edge_list],

        # Problem-specific objective value
        objective_name: float(objective_value),

        # Normalization information
        "penalty_high": max(penalty.values()),

        # Solver information
        "mip_solve_time": mip_solve_time,
        "solver_status": str(status),
        "termination_condition": str(termination)
    }

    # Shortest path and min-cost flow use arc costs.
    if cost is not None:
        sample["dist"] = [cost[u, v] for u, v in edge_list]
        sample["cost_high"] = max(cost.values())

    # Max flow and min-cost flow use capacities.
    if capacity is not None:
        sample["capacity"] = [capacity[u, v] for u, v in edge_list]
        sample["capacity_high"] = max(capacity.values())

    return sample



def build_dualILP(nodes, arcs, cost, supply, penalty, attack_limit=1):
    
    '''Constructs the dual formulation of the shortest path interdiction problem.

    Decision variables:
    - Pi[i]: node potentials (dual variables)
    - Y[i,j]: binary interdiction decision on arc (i,j)

    Objective:
    - Maximize shortest path length after interdiction'''
    
    # initialize model
    model = pyo.ConcreteModel()

    # initialize nodes and arcs as pyomo objects
    model.N = pyo.Set(initialize=list(nodes), ordered=True)
    model.A = pyo.Set(within=model.N * model.N, initialize=list(arcs))

    # initialize cost, penalty, supply and attack limits as pyomo objects
    model.cost = pyo.Param(model.A, initialize=cost)
    model.penalty = pyo.Param(model.A, initialize=penalty)
    model.supply = pyo.Param(model.N, initialize=supply)
    model.attack_limit = pyo.Param(initialize=attack_limit)

    # initialize decision variables described above
    model.Pi = pyo.Var(model.N, within=pyo.Reals)
    model.Y = pyo.Var(model.A, within=pyo.Binary)

    # dual feasibility constraint
    def dual_constraint_rule(model, i, j):
        return model.Pi[i] - model.Pi[j] <= model.cost[i, j] + model.penalty[i, j] * model.Y[i, j]
    model.dual_constraints = pyo.Constraint(model.A, rule=dual_constraint_rule)

    # initerdiction budget constraint
    def attack_limit_rule(model):
        return sum(model.Y[i, j] for (i, j) in model.A) <= model.attack_limit
    model.attack_budget = pyo.Constraint(rule=attack_limit_rule)

    # objective function = maximize the distance between s-t
    def objective_rule(model):
        return sum(model.supply[i] * model.Pi[i] for i in model.N if supply[i] != 0)
    model.pathLength = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    return model


def solve_shortest_path_instance(G, s, t, density, attack_limit):

    '''Solves one interdiction instance and returns a training sample.

   Returns:
   - graph structure (u, v, dist)
   - optimal interdiction decisions (Y)
   - resulting path length'''
   
   # ensures s-t path exists before solving
    if not nx.has_path(G, s, t):
        return None

    # initialize network data using function build_instance_data
    nodes, arcs, cost, _, penalty, supply = build_instance_data(G, s, t, problem_type="shortest_path")

    # use new data from build_instance_data to build the MIP using build_dualILP
    model = build_dualILP(nodes=nodes,arcs=arcs,cost=cost,supply=supply,
        penalty=penalty,attack_limit=attack_limit)

    # Solve MIP
    opt = pyo.SolverFactory('gurobi')

    # time the solve step
    solve_start = time.perf_counter()
    results = opt.solve(model)
    solve_end = time.perf_counter()

    # calculate solve time in seconds
    mip_solve_time = solve_end - solve_start
    
    status = results.solver.status
    termination = results.solver.termination_condition

    # Skip non-optimal solves
    if termination != pyo.TerminationCondition.optimal:
        print(f"Skipped: solver ended with {termination}")
        return None

    # created sorted edge list
    edge_list = sorted(arcs)

    return build_training_sample(G=G,s=s,t=t,density=density,attack_limit=attack_limit,arcs=arcs,
                                 penalty=penalty,model=model,objective_name="path_length",
                                 objective_value=pyo.value(model.pathLength),mip_solve_time=mip_solve_time,
                                 status=status,termination=termination,cost=cost)




def build_max_flow_ILP(nodes, arcs, capacity, penalty, source, sink, attack_limit=1):

    '''Constructs the dual formulation of the maximum-flow interdiction problem.

    Decision variables:
    - Pi[i]: node potentials
    - Alpha[i,j]: dual variables for arc capacities
    - AlphaReturn: dual variable for artificial return arc
    - Y[i,j]: binary interdiction decisions

    Objective:
    - Minimize the defender's maximum s-t flow after interdiction'''

    
    # initialize model
    model = pyo.ConcreteModel()

    # initialize nodes and arcs as pyomo objects
    model.N = pyo.Set(initialize=list(nodes), ordered=True)
    model.A = pyo.Set(within=model.N * model.N, initialize=list(arcs))

    # initialize cost, penalty, supply and attack limits as pyomo objects
    model.capacity = pyo.Param(model.A, initialize=capacity)
    model.penalty = pyo.Param(model.A, initialize=penalty)
    model.attack_limit = pyo.Param(initialize=attack_limit)

    # sufficiently large upper bound for artificial return arc
    max_capacity = max(capacity.values())
    return_capacity = len(nodes) * max_capacity

    model.return_capacity = pyo.Param(initialize=return_capacity)

    # initialize decision variables described above
    model.Pi = pyo.Var(model.N, within=pyo.Reals)
    model.Y = pyo.Var(model.A, within=pyo.Binary)
    model.Alpha = pyo.Var(model.A, within=pyo.NonNegativeReals)
    model.AlphaReturn = pyo.Var(within=pyo.NonNegativeReals)

    # dual feasibility constraint
    def dual_constraint_rule(model, i, j):
        return (model.Pi[i] - model.Pi[j] + model.Alpha[i,j] + model.penalty[i,j]*model.Y[i,j] >= 0)
    model.dual_constraints = pyo.Constraint(model.A, rule=dual_constraint_rule)

    # dual constraint for artificial return arc
    def return_arc_constraint_rule(model):
        return (model.Pi[sink] - model.Pi[source] + model.AlphaReturn >= 1)
    model.return_arc_constraint = pyo.Constraint(rule=return_arc_constraint_rule)

    # initerdiction budget constraint
    def attack_limit_rule(model):
        return sum(model.Y[i, j] for (i, j) in model.A) <= model.attack_limit
    model.attack_budget = pyo.Constraint(rule=attack_limit_rule)

    # anchor one node potential
    model.source_potential = pyo.Constraint(expr=model.Pi[source] == 0)

    # objective: minimize maximum remaining flow
    def objective_rule(model):
        return (sum(model.capacity[i,j]*model.Alpha[i,j] for i, j in model.A) 
                + model.return_capacity*model.AlphaReturn)
    model.maxFlow = pyo.Objective(rule=objective_rule,sense=pyo.minimize)

    return model


def solve_max_flow_instance(G, s, t, density, attack_limit):

    '''Solves one maximum-flow interdiction instance and returns a training sample.

    Returns:
    - graph structure (u, v, capacity)
    - optimal interdiction decisions (Y)
    - resulting maximum flow value
    '''

    # ensures s-t path exists before solving
    if not nx.has_path(G, s, t):
        return None

    # initialize network data using function build_instance_data
    nodes, arcs, _, capacity, penalty, _ = build_instance_data(G,s,t,problem_type="max_flow")

    # build the maximum-flow interdiction MIP
    model = build_max_flow_ILP(nodes=nodes,arcs=arcs,capacity=capacity,penalty=penalty,
                               source=s,sink=t,attack_limit=attack_limit)

    # solve MIP
    opt = pyo.SolverFactory("gurobi")

    # time the solve step
    solve_start = time.perf_counter()
    results = opt.solve(model)
    solve_end = time.perf_counter()

    # calculate solve time in seconds
    mip_solve_time = solve_end - solve_start

    status = results.solver.status
    termination = results.solver.termination_condition

    # skip non-optimal solves
    if termination != pyo.TerminationCondition.optimal:
        print(f"Skipped: solver ended with {termination}")
        return None

    return build_training_sample(G=G,s=s,t=t,density=density,attack_limit=attack_limit,
                                 arcs=arcs,penalty=penalty,model=model,objective_name="max_flow",
                                 objective_value=pyo.value(model.maxFlow),mip_solve_time=mip_solve_time,
                                 status=status,termination=termination,capacity=capacity)




def build_min_cost_flow_ILP(nodes, arcs, cost, capacity, supply, penalty, source, attack_limit=1):

    '''Constructs the dual formulation of the minimum-cost-flow interdiction problem.

    Decision variables:
    - Pi[i]: node potentials
    - Alpha[i,j]: dual variables for arc-capacity constraints
    - Y[i,j]: binary interdiction decisions

    Objective:
    - Maximize the defender's minimum flow cost after interdiction'''

    # initialize model
    model = pyo.ConcreteModel()

    # initialize nodes and arcs as Pyomo objects
    model.N = pyo.Set(initialize=list(nodes),ordered=True)

    model.A = pyo.Set(within=model.N * model.N,initialize=list(arcs))

    # initialize model parameters
    model.cost = pyo.Param(model.A,initialize=cost)
    model.capacity = pyo.Param(model.A,initialize=capacity)
    model.penalty = pyo.Param(model.A,initialize=penalty)
    model.supply = pyo.Param(model.N,initialize=supply)
    model.attack_limit = pyo.Param(initialize=attack_limit)

    # initialize decision variables
    model.Pi = pyo.Var(model.N,within=pyo.Reals)
    model.Alpha = pyo.Var(model.A, within=pyo.NonNegativeReals)
    model.Y = pyo.Var(model.A,within=pyo.Binary)

    # dual feasibility constraints
    def dual_constraint_rule(model, i, j):
        return (model.Pi[i] - model.Pi[j] - model.Alpha[i,j] - model.penalty[i,j]*model.Y[i,j] 
                <= model.cost[i, j])
    model.dual_constraints = pyo.Constraint(model.A,rule=dual_constraint_rule)

    # interdiction budget constraint
    def attack_limit_rule(model):
        return sum(model.Y[i,j] for i, j in model.A) <= model.attack_limit
    model.attack_budget = pyo.Constraint(rule=attack_limit_rule)
    
    model.source_potential = pyo.Constraint(expr=model.Pi[source] == 0)

    # maximize the minimum flow cost after interdiction
    def objective_rule(model):
        return (sum(model.supply[i]*model.Pi[i] for i in model.N) - 
                sum(model.capacity[i,j]*model.Alpha[i,j] for i, j in model.A))
    model.minCostFlow = pyo.Objective(rule=objective_rule,sense=pyo.maximize)

    return model


def solve_min_cost_flow_instance(G, s, t, density, attack_limit):

    '''Solves one minimum-cost-flow interdiction instance and returns a training sample.

    Returns:
    - graph structure (u, v, dist, capacity)
    - optimal interdiction decisions (Y)
    - resulting minimum-cost-flow objective value
    '''

    # ensures s-t path exists before solving
    if not nx.has_path(G, s, t):
        return None

    # initialize network data using function build_instance_data
    nodes, arcs, cost, capacity, penalty, supply = build_instance_data(G,s,t,
                                                                       problem_type="min_cost_flow")

    # build the minimum-cost-flow interdiction MIP
    model = build_min_cost_flow_ILP(nodes=nodes,arcs=arcs,cost=cost,capacity=capacity,supply=supply,
                                    penalty=penalty,source=s,attack_limit=attack_limit)

    # solve MIP
    opt = pyo.SolverFactory("gurobi")

    # time the solve step
    solve_start = time.perf_counter()
    results = opt.solve(model)
    solve_end = time.perf_counter()

    # calculate solve time in seconds
    mip_solve_time = solve_end - solve_start

    status = results.solver.status
    termination = results.solver.termination_condition

    # skip non-optimal solves
    if termination != pyo.TerminationCondition.optimal:
        print(f"Skipped: solver ended with {termination}")
        return None

    return build_training_sample(G=G,s=s,t=t,density=density,attack_limit=attack_limit,arcs=arcs,
                                 penalty=penalty,model=model,objective_name="min_cost_flow",
                                 objective_value=pyo.value(model.minCostFlow),mip_solve_time=mip_solve_time,
                                 status=status,termination=termination,cost=cost,capacity=capacity)




def solve_instance(G,s,t,density,attack_limit,problem_type="shortest_path"):

    """Send an instance to the solver matching its problem type."""

    if problem_type == "shortest_path":
        return solve_shortest_path_instance(G=G,s=s,t=t,density=density,attack_limit=attack_limit)

    if problem_type == "max_flow":
        return solve_max_flow_instance(G=G,s=s,t=t,density=density,attack_limit=attack_limit)

    if problem_type == "min_cost_flow":
        return solve_min_cost_flow_instance(G=G,s=s,t=t,density=density,attack_limit=attack_limit)

    raise ValueError(
        f"Unknown problem_type: {problem_type}. "
        "Expected 'shortest_path', 'max_flow', or 'min_cost_flow'.")