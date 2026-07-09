# -*- coding: utf-8 -*-
"""
Created on Sat May 30 05:36:20 2026

@author: emmallen
"""

import networkx as nx
import pyomo.environ as pyo
import time


def build_instance_data(G, s, t, interdiction_penalty=1, problem_type="shortest_path"):
    
    '''Convert graph into sets and parameters for the MIP model.

   Returns:
   - nodes: list of nodes
   - arcs: list of directed edges
   - cost: arc length dictionary
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

    if problem_type == "shortest_path":

        # arc distances / costs
        cost = {(u, v): G[u][v]["dist"] for (u, v) in arcs}

        return nodes, arcs, cost, penalty, supply
    
    elif problem_type == "max_flow":

        # arc capacities
        capacity = {(u, v): G[u][v]["capacity"] for (u, v) in arcs}

        return nodes, arcs, capacity, penalty, supply
    
    elif problem_type == "min_cost_flow":

        # per-unit flow costs
        cost = {(u, v): G[u][v]["dist"] for (u, v) in arcs}

        # arc capacities
        capacity = {(u, v): G[u][v]["capacity"] for (u, v) in arcs}

        return nodes, arcs, cost, capacity, penalty, supply



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


def solve_shortest_path_instance(G, s, t, density, attack_limit,interdiction_penalty=1,
                                 problem_type="shortest_path"):
    
    '''Solves one interdiction instance and returns a training sample.

   Returns:
   - graph structure (u, v, dist)
   - optimal interdiction decisions (Y)
   - resulting path length'''
   
   # ensures s-t path exists before solving
    if not nx.has_path(G, s, t):
        return None

    # initialize network dat a using function build_instance_data
    nodes, arcs, cost, penalty, supply = build_instance_data(G, s, t, interdiction_penalty)

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

    # cosntruct training sample
    sample = {
        # graph structure
        "n_nodes": G.number_of_nodes(),
        "n_arcs": G.number_of_edges(),
        "density": density,
        "source": s,
        "sink": t,
        "attack_limit": attack_limit,
        "penalty": [penalty[(u, v)] for (u, v) in edge_list],
        "u": [u for (u, v) in edge_list],
        "v": [v for (u, v) in edge_list],
        "dist": [cost[(u, v)] for (u, v) in edge_list],
        
        # what model is supposed to learn (attack values)
        "attack": [int(round(pyo.value(model.Y[u, v]))) for (u, v) in edge_list],
        # target value = shortest path
        "path_length": float(pyo.value(model.pathLength)),

        "cost_high": max(cost.values()),
        "penalty_high": max(penalty.values()),
        
        # additional info for analysis
        "mip_solve_time": mip_solve_time,
        "solver_status": str(status),
        "termination_condition": str(termination)}

    return sample