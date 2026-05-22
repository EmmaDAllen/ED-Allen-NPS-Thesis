# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:42:40 2026

@author: emmallen
"""

'''Training set generation for shortest path interdiction thesis.

1. Imports One-In random networks from random_test_networks.py
2. Solves each generated network using the MIP formulation
3. Saves the resulting training data to JSON'''


import networkx as nx
import pyomo.environ as pyo
import json

from random_test_networks import generate_one_in_network 
from random_test_networks import generate_grid_network
from random_test_networks import generate_spatial_network
from random_test_networks import generate_hub_spoke_network


def choose_source_sink(G):
    
    '''Function defines source and sink nodes.'''
    
    # source node = 0
    s = 0
    # sink node = last node created
    t = G.number_of_nodes() - 1
    
    return s, t


def build_instance_data(G, s, t):
    
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

    # edge weights = shortest path costs
    cost = {(u, v): G[u][v]["dist"] for (u, v) in arcs}
    # interdiction penalty = currently set to 1
    penalty = {(u, v): 1 for (u, v) in arcs}

    # flow balance supply values
    supply = {i: 0 for i in nodes}
    supply[s] = 1
    supply[t] = -1

    return nodes, arcs, cost, penalty, supply


def build_dualILP(nodes, arcs, cost, supply, penalty, attack_limit):
    
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


def solve_instance(G, s, t, density, max_attacks=1):
    
    '''Solves one interdiction instance and returns a training sample.

   Returns:
   - graph structure (u, v, dist)
   - optimal interdiction decisions (Y)
   - resulting path length'''
   
   # ensures s-t path exists before solving
    if not nx.has_path(G, s, t):
        return None

    # initialize network dat a using function build_instance_data
    nodes, arcs, cost, penalty, supply = build_instance_data(G, s, t)

    # use new data from build_instance_data to build the MIP using build_dualILP
    model = build_dualILP(nodes=nodes,arcs=arcs,cost=cost,supply=supply,
        penalty=penalty,attack_limit=max_attacks)

    # Solve MIP
    opt = pyo.SolverFactory('gurobi_direct')
    results = opt.solve(model)

    
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
        "attack_limit": max_attacks,
        "u": [u for (u, v) in edge_list],
        "v": [v for (u, v) in edge_list],
        "dist": [cost[(u, v)] for (u, v) in edge_list],
        
        # what model is supposed to learn (attack values)
        "attack": [int(round(pyo.value(model.Y[u, v]))) for (u, v) in edge_list],
        # target value = shortest path
        "path_length": float(pyo.value(model.pathLength)),
        
        "solver_status": str(status),
        "termination_condition": str(termination)}

    return sample


def generate_dataset(network_settings,replications_per_setting, max_attacks=1,
    base_seed=1,output_file="training_data.json"):
    
    '''Generates dataset across multiple network sizes and densities.

    For each (n, m):
        - generate multiple random networks
        - solve each using MIP
        - store results

    Output:
        JSON file containing training samples'''
    
    
    dataset = []
    skipped = 0

    for n, m in network_settings:
        for rep in range(replications_per_setting):
            
            # unique seed per instance (ensures reproducibility)
            seed = base_seed + 100000 * n + 100 * m + rep

            # generate random test network (One-In method)
            G, s, t, density = generate_one_in_network(n=n, m=m,cost_low=1,
                cost_high=10,seed=seed)

            #G, s, t, density = generate_spatial_network(n=n,k=4,seed=seed)
            
            #G, s, t, density = generate_grid_network(rows=10,cols=10,seed=seed)
            
            #G, s, t, density = generate_hub_spoke_network(n=n,num_hubs=5,seed=seed)
            
            # solve interdiction problem
            sample = solve_instance(G=G,s=s,t=t,density=density,max_attacks=max_attacks)

            if sample is None:
                skipped += 1
                continue

            sample["graph_seed"] = seed
            sample["replication"] = rep
            dataset.append(sample)

            print(f"Solved n={n}, m={m}, density={density:.2f}, "
                f"rep={rep}, objective={sample['path_length']:.2f}")

    print(f"\nGenerated {len(dataset)} solved training samples.")
    print(f"Skipped {skipped} instances.")

    # save dataset
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset


if __name__ == "__main__":

    # experiment design: (n, m) pairs
    network_settings = [
        (30, 75),
        (30, 120),
        (30, 180),
        (50, 125),
        (50, 200),
        (50, 300),
        (75, 188),
        (75, 300),
        (75, 450),
    ]

    dataset = generate_dataset(
        network_settings=network_settings,
        replications_per_setting=100,
        max_attacks=1,
        base_seed=1,
        output_file="training_data.json")
