#!/usr/bin/env python3

'''
Solving Combinatorial Optimization with QAOA

qiskit tutorial: https://qiskit.org/textbook/ch-applications/qaoa.html
'''

## Summary
# MAXCUT problem on butterfly graph of (IBMQ) 5-qubit chip.
# Graph corresponds to native connectivity of chip, so cost function and
#  Hamiltonian coincide.
# Cost function can be analytically calculated, avoiding need to find optimal
#  parameters variationally.


# Import necessary tools
print("Importing tools...")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
print("Tools imported.")


def show_plot():
    '''
    Maximizes current plot.
    '''
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()


# Generates butterfly graph to represent 5-qubit chip connectivity.
n_qubits = 5
nodes = np.arange(0, n_qubits)
edges = [(0, 1), (0, 2), (1, 2), (3, 2), (3, 4), (4, 2)]
edge_weight = 1
weighted_edges = [edge + (edge_weight,) for edge in edges]
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_weighted_edges_from(weighted_edges)


# Visualizes plot of graph.
colors = ['r' for node in graph.nodes()]
transparency = 1
default_axes = plt.axes(frameon=True)
position = nx.spring_layout(graph)
nx.draw_networkx(graph, node_color=colors, node_size=600, alpha=transparency,
                 ax=default_axes, pos=position)
show_plot()


# Maximize expectation for selected trial state, to be simulated.
step_size = 0.1


if __name__ == "__main__":
    pass
