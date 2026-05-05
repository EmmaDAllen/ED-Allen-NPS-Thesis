# Tropical Attention for Network Interdiction

**BLUF:** A learning-based framework that approximates shortest-path network interdiction solutions using tropical attention, reducing reliance on computationally expensive MIP solvers.


## Abstract

Network interdiction identifies critical components whose disruption maximizes system impact and is widely used to analyze vulnerabilities in military logistics, transportation, and infrastructure networks within the Department of Defense. These problems are typically formulated as bilevel optimization models and become computationally expensive to solve using exact methods such as mixed-integer programming (MIP), particularly as network size increases.

This thesis develops a learning-based framework to approximate solutions to shortest-path network interdiction problems. MIP is used to generate interdiction instances and provide ground truth solutions for training. A tropical attention-based neural network is developed to learn combinatorial patterns in shortest-path behavior by leveraging max-plus structure and the dynamic programming nature of shortest-path computations.

The model uses features derived from network topology, arc weights, and source-sink structure to predict interdiction decisions on new instances. Computational experiments evaluate model performance against MIP solutions using objective value gap, decision accuracy, Hamming distance, and runtime. An extension incorporating structural features from Benders decomposition is also explored to improve performance. This work supports scalable network interdiction analysis and faster evaluation of network vulnerabilities.


## Problem Statement

Exact methods for network interdiction (e.g., MIP, Benders decomposition) become computationally intractable for large-scale networks. This limits the ability to rapidly evaluate multiple network scenarios, which is critical for operational decision-making.

This project investigates whether a learning-based approach can approximate optimal interdiction decisions and objective values with significantly reduced runtime.


## Approach

- Generate random directed networks using NetworkX
- Solve shortest-path interdiction instances using MIP (ground truth)
- Train a tropical attention neural network to learn:
  - Interdiction decisions (binary arc selection)
  - Resulting shortest-path objective value
- Evaluate model performance on unseen networks

### Extension
- Incorporate structural features derived from Benders decomposition:
  - Arc criticality
  - Path participation
  - Cut-based sensitivity
 

## Repository Structure

├── data/ # Generated network instances
├── models/ # Trained model checkpoints
├── notebooks/ # Experimentation and analysis
├── src/ # Core code (data generation, MIP, training)
├── results/ # Evaluation outputs and figures
├── README.md



## Setup


## Usage


## Evaluation Metrics


## Results


## Future Work


## References
