# -*- coding: utf-8 -*-
"""
Created on Sat May 30 05:36:20 2026

@author: emmallen
"""

import networkx as nx
import pyomo.environ as pyo
import time


def build_instance_data(G,s,t,problem_type="shortest_path",flow_demand=1):
    
    """Convert a NetworkX graph into sets and parameter dictionaries.

    Returns:
        nodes: List of graph nodes.
        arcs: List of directed arcs.
        cost: Arc-cost dictionary when required; otherwise None.
        capacity: Arc-capacity dictionary when required; otherwise None.
        penalty: Arc interdiction-penalty dictionary.
        supply: Node supply/demand dictionary when required; otherwise None."""
    
    # if problem type is not recognized, raise an error
    if problem_type not in ["shortest_path", "max_flow", "min_cost_flow"]:
        raise ValueError(f"Unknown problem_type: {problem_type}")
   
   # list of nodes
    nodes = list(G.nodes())
    # list of edges
    arcs = list(G.edges())

    # interdiction penalty = currently set to 1
    penalty = {(u, v): G[u][v]["penalty"] for (u, v) in arcs}

    # initialize cost, capacity, and supply to None; will be set based on problem type
    cost = None
    capacity = None
    supply = None

    # if problem type is shortest path or min-cost flow, set cost and supply
    if problem_type in ["shortest_path", "min_cost_flow"]:
        cost = {(u, v): G[u][v]["dist"] for (u, v) in arcs} # use arc distances as costs
        supply = {i: 0 for i in nodes} # initialize supply/demand to zero for all nodes
        supply[s] = flow_demand # source node has supply equal to flow_demand
        supply[t] = -flow_demand # sink node has demand equal to flow_demand

    # if problem type is max flow or min-cost flow, set capacity
    if problem_type in ["max_flow", "min_cost_flow"]:
        capacity = {(u, v): G[u][v]["capacity"] for (u, v) in arcs} # use arc capacities

    return nodes, arcs, cost, capacity, penalty, supply




def build_training_sample(G,s,t,density,attack_limit,arcs,penalty,model,objective_name,objective_value,
                          mip_solve_time,status,termination,problem_type,cost=None,capacity=None,flow_demand=None, 
                          baseline_max_flow=None):

    """Converts the solved model into one dictionary that is saved as JSON"""
    
    # created sorted edge list
    edge_list = sorted(arcs)

    # construct training sample
    sample = {
        "n_nodes": G.number_of_nodes(), # number of nodes
        "n_arcs": G.number_of_edges(), # number of edges
        "density": density, # experimental density = arc to node ratio
        "source": s, # source node
        "sink": t, # sink node
        "attack_limit": attack_limit, # attack limit
        "problem_type": problem_type, # problem type

        # arc information
        "u": [u for u, v in edge_list], # arc heads
        "v": [v for u, v in edge_list], # arc tails
        "penalty": [penalty[u, v] for u, v in edge_list], # arc penalties

        # optimal interdiction decisions
        "attack": [int(round(pyo.value(model.Y[u, v]))) for u, v in edge_list],

        # problem-specific objective value
        objective_name: float(objective_value),

        # stores the largest penalty that actually appeared in each graph
        "penalty_high": max(penalty.values()),

        # Solver information
        "mip_solve_time": mip_solve_time,
        "solver_status": str(status),
        "termination_condition": str(termination)
    }

    # shortest path and min-cost flow use arc costs
    # stores the cost of each arc and the largest cost that actually appeared in each graph
    if cost is not None:
        sample["dist"] = [cost[u, v] for u, v in edge_list]
        sample["cost_high"] = max(cost.values())

    # max flow and min-cost flow use capacities
    # stores the capacity of each arc and the largest capacity that actually appeared in each graph
    if capacity is not None:
        sample["capacity"] = [capacity[u, v] for u, v in edge_list]
        sample["capacity_high"] = max(capacity.values())

    # min-cost flow uses flow demand and baseline max flow
    if flow_demand is not None:
        sample["flow_demand"] = flow_demand

    # baseline max flow is only relevant for max flow and min-cost flow problems
    if baseline_max_flow is not None:
        sample["baseline_max_flow"] = float(baseline_max_flow)

    return sample




def build_shortest_pathILP(nodes,arcs,cost,supply,penalty,attack_limit=1):
    
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
    model.A = pyo.Set(within=model.N*model.N, initialize=list(arcs))

    # initialize cost, penalty, supply and attack limits as pyomo objects
    model.cost = pyo.Param(model.A, initialize=cost)
    model.penalty = pyo.Param(model.A, initialize=penalty)
    model.supply = pyo.Param(model.N, initialize=supply)
    model.attack_limit = pyo.Param(initialize=attack_limit)

    # initialize decision variables described above
    model.Pi = pyo.Var(model.N, within=pyo.Reals)
    model.Y = pyo.Var(model.A, within=pyo.Binary)

    # dual feasibility constraint
    def dual_constraint_rule(model,i,j):
        return model.Pi[i] - model.Pi[j] - model.penalty[i,j]*model.Y[i,j] <= model.cost[i,j]
    model.dual_constraints = pyo.Constraint(model.A, rule=dual_constraint_rule)

    # initerdiction budget constraint
    def attack_limit_rule(model):
        return sum(model.Y[i,j] for (i,j) in model.A) <= model.attack_limit
    model.attack_budget = pyo.Constraint(rule=attack_limit_rule)

    # objective function = maximize the distance between s-t 
    # supply[s] = 1, supply[t] = -1, supply[i] = 0 for all other nodes, so the objective is equivalent to maximizing Pi[s] - Pi[t]
    def objective_rule(model):
        return sum(model.supply[i]*model.Pi[i] for i in model.N if supply[i] != 0)
    model.pathLength = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    return model


def solve_shortest_path_instance(G, s, t, density, attack_limit):

    '''Solves one interdiction instance and returns a training sample.

   Returns:
   - graph structure (u, v, dist)
   - optimal interdiction decisions (Y)
   - resulting path length'''
   
   # ensures s-t path exists before solving
    if not nx.has_path(G,s,t):
        return None

    # initialize network data using function build_instance_data
    nodes, arcs, cost, _, penalty, supply = build_instance_data(G,s,t,problem_type="shortest_path")

    # use new data from build_instance_data to build the MIP using build_dualILP
    model = build_shortest_pathILP(nodes=nodes,arcs=arcs,cost=cost,supply=supply,
        penalty=penalty,attack_limit=attack_limit)

    # Solve MIP using gurobi solver
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

    return build_training_sample(G=G,s=s,t=t,density=density,attack_limit=attack_limit,arcs=arcs,
                                 penalty=penalty,model=model,objective_name="path_length",
                                 objective_value=pyo.value(model.pathLength),mip_solve_time=mip_solve_time,
                                 status=status,termination=termination,problem_type="shortest_path",cost=cost)




def build_max_flow_ILP(nodes,arcs,capacity,source,sink,attack_limit=1):

    """Construct a maximum-flow interdiction model using a minimum-cut formulation.

    Decision variables:
        X[i]: 1 if node i is on the source side of the cut.
        Y[i,j]: 1 if arc (i,j) is interdicted.
        Z[i,j]: 1 if arc (i,j) survives and crosses the cut.

    Objective:
        Minimize the capacity of the surviving source-sink cut."""

    # initialize model
    model = pyo.ConcreteModel()

    # initialize nodes and arcs as pyomo objects
    model.N = pyo.Set(initialize=list(nodes), ordered=True)
    model.A = pyo.Set(within=model.N * model.N, initialize=list(arcs))

    # initialize cost, penalty, supply and attack limits as pyomo objects
    model.capacity = pyo.Param(model.A, initialize=capacity)
    model.attack_limit = pyo.Param(initialize=attack_limit)

    # Node side of the s-t cut - 1 = source side, 0 = sink side
    model.X = pyo.Var(model.N,within=pyo.Binary)
    # Interdiction decisions
    model.Y = pyo.Var(model.A,within=pyo.Binary)
    # Surviving arcs crossing the cut
    model.Z = pyo.Var(model.A,within=pyo.Binary)

    # Source and sink must be on opposite sides
    model.source_side = pyo.Constraint(expr=model.X[source] == 1)
    model.sink_side = pyo.Constraint(expr=model.X[sink] == 0)

    # Surviving arcs must cross the min-cut if they are not interdicted
    # arc crosses the cut and is not interdicted => Z[i,j] = 1 = capcaity is counted in the objective
    # arc crosses the cut and is interdicted => Z[i,j] = 0 = capacity is not counted in the objective
    # arc does not cross the cut => Z[i,j] = 0 = capacity is not counted in the objective
    def cut_arc_rule(model,i,j):
        return (model.Z[i,j] >= model.X[i] - model.X[j] - model.Y[i,j])
    model.cut_arc_constraints = pyo.Constraint(model.A,rule=cut_arc_rule)

    # initerdiction budget constraint
    def attack_limit_rule(model):
        return sum(model.Y[i,j] for i, j in model.A) <= model.attack_limit
    model.attack_budget = pyo.Constraint(rule=attack_limit_rule)

    # adds capacity of surviving arcs crossing the cut to the objective function
    # minimum cut capacity = post-interdiction maximum flow value
    model.maxFlow = pyo.Objective(expr=sum(model.capacity[i,j] * model.Z[i,j] for i,j in model.A),sense=pyo.minimize)

    return model




def solve_max_flow_instance(G,s,t,density, attack_limit):

    '''Solves one maximum-flow interdiction instance and returns a training sample.

    Returns:
    - graph structure (u, v, capacity)
    - optimal interdiction decisions (Y)
    - resulting maximum flow value'''

    # ensures s-t path exists before solving
    if not nx.has_path(G, s, t):
        return None

    # initialize network data using function build_instance_data
    nodes, arcs, _, capacity, penalty, _ = build_instance_data(G,s,t,problem_type="max_flow")

    baseline_max_flow = nx.maximum_flow_value(G,s,t,capacity="capacity")

    # build the maximum-flow interdiction MIP
    model = build_max_flow_ILP(nodes=nodes,arcs=arcs,capacity=capacity,source=s,sink=t,
                               attack_limit=attack_limit)
    
    # solve MIP using gurobi solver
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
    

    return build_training_sample(G=G,s=s,t=t,density=density,attack_limit=attack_limit,arcs=arcs,penalty=penalty,model=model,
                                 objective_name="max_flow",objective_value=pyo.value(model.maxFlow),mip_solve_time=mip_solve_time,
                                 status=status,termination=termination,problem_type="max_flow",capacity=capacity,
                                 baseline_max_flow=baseline_max_flow)





def build_min_cost_flow_ILP(nodes,arcs,cost,capacity,supply,penalty,source,attack_limit=1):

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
    def dual_constraint_rule(model,i,j):
        return (model.Pi[i] - model.Pi[j] - model.Alpha[i,j] - model.penalty[i,j]*model.Y[i,j] 
                <= model.cost[i,j])
    model.dual_constraints = pyo.Constraint(model.A,rule=dual_constraint_rule)

    # interdiction budget constraint
    def attack_limit_rule(model):
        return sum(model.Y[i,j] for i, j in model.A) <= model.attack_limit
    model.attack_budget = pyo.Constraint(rule=attack_limit_rule)
    
    # maximize the minimum flow cost after interdiction
    def objective_rule(model):
        return (sum(model.supply[i]*model.Pi[i] for i in model.N) - 
                sum(model.capacity[i,j]*model.Alpha[i,j] for i, j in model.A))
    model.minCostFlow = pyo.Objective(rule=objective_rule,sense=pyo.maximize)

    return model




def solve_min_cost_flow_instance(G, s, t, density, attack_limit,flow_demand):

    '''Solves one minimum-cost-flow interdiction instance and returns a training sample.

    Returns:
    - graph structure (u, v, dist, capacity)
    - optimal interdiction decisions (Y)
    - resulting minimum-cost-flow objective value
    '''

    # ensures s-t path exists before solving
    if not nx.has_path(G, s, t):
        return None
    
    baseline_max_flow = nx.maximum_flow_value(G,s,t,capacity="capacity",)

    if baseline_max_flow < flow_demand:
        return None

    # initialize network data using function build_instance_data
    nodes, arcs, cost, capacity, penalty, supply = build_instance_data(G,s,t,
                                                                       problem_type="min_cost_flow",
                                                                       flow_demand=flow_demand)

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

    return build_training_sample(G=G,s=s,t=t,density=density,attack_limit=attack_limit,arcs=arcs,penalty=penalty,model=model,
                                 objective_name="min_cost_flow",objective_value=pyo.value(model.minCostFlow),mip_solve_time=mip_solve_time,
                                 status=status,termination=termination,problem_type="min_cost_flow",cost=cost,capacity=capacity,
                                 flow_demand=flow_demand)




def solve_instance(G,s,t,density,attack_limit,problem_type="shortest_path", flow_demand=1):

    """Send an instance to the solver matching its problem type."""

    if problem_type == "shortest_path":
        return solve_shortest_path_instance(G=G,s=s,t=t,density=density,attack_limit=attack_limit)

    if problem_type == "max_flow":
        return solve_max_flow_instance(G=G,s=s,t=t,density=density,attack_limit=attack_limit)

    if problem_type == "min_cost_flow":
        return solve_min_cost_flow_instance(G=G,s=s,t=t,density=density,attack_limit=attack_limit,flow_demand=flow_demand)

    raise ValueError(
        f"Unknown problem_type: {problem_type}. "
        "Expected 'shortest_path', 'max_flow', or 'min_cost_flow'.")