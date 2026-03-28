from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ============================================================
# PageRank Implementation and Validation
# Dataset: web-Google_10k.txt
# Methods:
#   1. Power iteration on the full graph
#   2. Closed-form validation on a reduced subgraph
# ============================================================

# -----------------------------
# Experiment parameters
# -----------------------------
DATASET_PATH = "web-Google_10k.txt"
TELEPORT_PROB = 0.15
EPSILON = 1e-6
MAX_ITER = 200
TOP_K = 10
SUBGRAPH_SIZE = 500


# -----------------------------
# Graph loading
# -----------------------------
def load_directed_graph(filepath: str) -> nx.DiGraph:
    """
    Load a directed graph from an edge-list file.

    Expected format:
    - Each valid line contains two entries: source target
    - Lines beginning with '#' are treated as comments
    - Blank lines are ignored
    """
    graph = nx.DiGraph()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            source, target = parts[0], parts[1]
            graph.add_edge(source, target)

    return graph


# -----------------------------
# Node relabeling
# -----------------------------
def relabel_to_contiguous_indices(
    graph: nx.DiGraph,
) -> Tuple[nx.DiGraph, Dict[str, int], Dict[int, str]]:
    """
    Relabel graph nodes to contiguous integer indices in [0, n-1].

    Returns:
    - relabeled graph
    - mapping from original node ID to integer index
    - mapping from integer index to original node ID
    """
    original_nodes = list(graph.nodes())
    original_to_new = {node: idx for idx, node in enumerate(original_nodes)}
    new_to_original = {idx: node for node, idx in original_to_new.items()}

    relabeled_graph = nx.relabel_nodes(graph, original_to_new, copy=True)
    return relabeled_graph, original_to_new, new_to_original


# -----------------------------
# Outgoing adjacency structure
# -----------------------------
def build_outgoing_lists(graph: nx.DiGraph) -> Tuple[List[List[int]], np.ndarray]:
    """
    Construct outgoing adjacency lists and out-degree counts.

    For each node j, outgoing_lists[j] stores the nodes reached by
    outgoing links from j.
    """
    n = graph.number_of_nodes()
    outgoing_lists: List[List[int]] = [[] for _ in range(n)]
    out_degree = np.zeros(n, dtype=np.int64)

    for node in range(n):
        neighbors = list(graph.successors(node))
        outgoing_lists[node] = neighbors
        out_degree[node] = len(neighbors)

    return outgoing_lists, out_degree


# -----------------------------
# Power iteration PageRank
# -----------------------------
def pagerank_power_iteration(
    graph: nx.DiGraph,
    p: float = 0.15,
    epsilon: float = 1e-6,
    max_iter: int = 200,
) -> Tuple[np.ndarray, int, float]:
    """
    Compute PageRank using power iteration without explicitly forming
    the transition matrix.

    Update rule:
        pi^(t+1) = (1-p) P pi^(t) + (p/n) 1

    Dangling nodes are handled by redistributing their probability mass
    uniformly across all nodes.
    """
    n = graph.number_of_nodes()
    outgoing_lists, out_degree = build_outgoing_lists(graph)

    pi = np.full(n, 1.0 / n, dtype=np.float64)

    for iteration in range(1, max_iter + 1):
        new_pi = np.full(n, p / n, dtype=np.float64)
        dangling_mass = 0.0

        for j in range(n):
            if out_degree[j] == 0:
                dangling_mass += pi[j]
            else:
                share = (1.0 - p) * pi[j] / out_degree[j]
                for i in outgoing_lists[j]:
                    new_pi[i] += share

        if dangling_mass > 0:
            new_pi += (1.0 - p) * dangling_mass / n

        new_pi /= new_pi.sum()
        l1_diff = np.linalg.norm(new_pi - pi, ord=1)

        if l1_diff < epsilon:
            return new_pi, iteration, l1_diff

        pi = new_pi

    return pi, max_iter, l1_diff


# -----------------------------
# Dense transition matrix
# -----------------------------
def build_dense_transition_matrix(graph: nx.DiGraph) -> np.ndarray:
    """
    Build the dense column-stochastic transition matrix P.

    P[i, j] = 1 / d_j if page j links to page i, and 0 otherwise.
    Dangling columns are replaced by the uniform distribution 1/n.
    """
    n = graph.number_of_nodes()
    P = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        neighbors = list(graph.successors(j))
        d_j = len(neighbors)

        if d_j == 0:
            P[:, j] = 1.0 / n
        else:
            for i in neighbors:
                P[i, j] = 1.0 / d_j

    return P


# -----------------------------
# Closed-form PageRank
# -----------------------------
def pagerank_closed_form(graph: nx.DiGraph, p: float = 0.15) -> np.ndarray:
    """
    Compute PageRank using the closed-form linear system:

        pi = (p/n) * (I - (1-p)P)^(-1) * 1

    The implementation uses np.linalg.solve rather than forming the
    explicit matrix inverse.
    """
    n = graph.number_of_nodes()
    P = build_dense_transition_matrix(graph)

    I = np.eye(n, dtype=np.float64)
    ones = np.ones(n, dtype=np.float64)

    A = I - (1.0 - p) * P
    b = (p / n) * ones

    pi = np.linalg.solve(A, b)
    pi /= pi.sum()

    return pi


# -----------------------------
# Validation subgraph extraction
# -----------------------------
def extract_validation_subgraph(
    graph: nx.DiGraph,
    subgraph_size: int = 500,
) -> nx.DiGraph:
    """
    Extract an induced subgraph for closed-form validation.

    The default procedure selects the first 'subgraph_size' nodes and
    constructs the induced subgraph on that set.
    """
    selected_nodes = list(graph.nodes())[:subgraph_size]
    subgraph = graph.subgraph(selected_nodes).copy()
    return subgraph


# -----------------------------
# Reporting utilities
# -----------------------------
def top_k_pagerank(
    pi: np.ndarray,
    new_to_original: Dict[int, str],
    k: int = 10,
) -> List[Tuple[int, str, float]]:
    """
    Return the top-k nodes ranked by PageRank.

    Output format:
    (rank, original_node_id, score)
    """
    ranked_indices = np.argsort(-pi)[:k]
    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        original_id = new_to_original[idx]
        results.append((rank, original_id, float(pi[idx])))

    return results



def compare_vectors(pi_iter: np.ndarray, pi_closed: np.ndarray) -> Dict[str, float]:
    """
    Compute numerical differences between iterative and closed-form
    PageRank vectors.
    """
    return {
        "L1_difference": float(np.linalg.norm(pi_iter - pi_closed, ord=1)),
        "L2_difference": float(np.linalg.norm(pi_iter - pi_closed, ord=2)),
        "max_absolute_difference": float(np.max(np.abs(pi_iter - pi_closed))),
        "sum_iterative": float(pi_iter.sum()),
        "sum_closed_form": float(pi_closed.sum()),
    }



def print_top_k_table(title: str, rows: List[Tuple[int, str, float]]) -> None:
    """Print a formatted top-k PageRank table."""
    print(title)
    print("-" * len(title))
    print(f"{'Rank':<6}{'Node ID':<20}{'PageRank Score'}")
    for rank, node_id, score in rows:
        print(f"{rank:<6}{node_id:<20}{score:.10f}")
    print()


# -----------------------------
# Distribution plots
# -----------------------------
def plot_pagerank_distribution(pi: np.ndarray) -> None:
    """
    Plot the empirical distribution of PageRank scores on the full graph.

    Figures produced:
    1. Histogram on a linear frequency scale
    2. Histogram with log-scaled frequency
    3. Empirical cumulative distribution function
    """
    plt.figure(figsize=(8, 5))
    plt.hist(pi, bins=50, edgecolor="black")
    plt.title("PageRank Distribution on Full Graph (Linear Scale)")
    plt.xlabel("PageRank Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(pi, bins=50, edgecolor="black")
    plt.yscale("log")
    plt.title("PageRank Distribution on Full Graph (Log-Scaled Frequency)")
    plt.xlabel("PageRank Score")
    plt.ylabel("Frequency (log scale)")
    plt.tight_layout()
    plt.show()

    sorted_pi = np.sort(pi)
    cdf = np.arange(1, len(sorted_pi) + 1) / len(sorted_pi)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_pi, cdf)
    plt.title("Cumulative Distribution of PageRank Scores")
    plt.xlabel("PageRank Score")
    plt.ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main experiment pipeline
# -----------------------------
def main() -> None:
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path.resolve()}\n"
            "Update DATASET_PATH to the correct file location before running the script."
        )

    print("=" * 70)
    print("PageRank Experiment")
    print("=" * 70)

    print(f"Loading graph from: {dataset_path}")
    raw_graph = load_directed_graph(str(dataset_path))
    graph, original_to_new, new_to_original = relabel_to_contiguous_indices(raw_graph)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    print(f"Number of nodes: {n}")
    print(f"Number of edges: {m}")
    print()

    print("Running PageRank (power iteration) on full graph...")
    start_time = time.time()
    pi_full, iterations, final_l1 = pagerank_power_iteration(
        graph=graph,
        p=TELEPORT_PROB,
        epsilon=EPSILON,
        max_iter=MAX_ITER,
    )
    full_runtime = time.time() - start_time

    print(f"Converged in iterations: {iterations}")
    print(f"Final L1 difference     : {final_l1:.10e}")
    print(f"Sum of PageRank scores  : {pi_full.sum():.12f}")
    print(f"Runtime (seconds)       : {full_runtime:.4f}")
    print()

    top_full = top_k_pagerank(pi_full, new_to_original, k=TOP_K)
    print_top_k_table("Top 10 Pages by PageRank (Full Graph)", top_full)

    print("Generating PageRank distribution charts for full graph...")
    plot_pagerank_distribution(pi_full)

    print(
        f"Extracting reduced subgraph for closed-form validation "
        f"(size={SUBGRAPH_SIZE})..."
    )
    raw_subgraph = extract_validation_subgraph(raw_graph, subgraph_size=SUBGRAPH_SIZE)
    subgraph, sub_o2n, sub_n2o = relabel_to_contiguous_indices(raw_subgraph)

    sub_n = subgraph.number_of_nodes()
    sub_m = subgraph.number_of_edges()

    print(f"Validation subgraph nodes: {sub_n}")
    print(f"Validation subgraph edges: {sub_m}")
    print()

    print("Running PageRank (power iteration) on validation subgraph...")
    pi_iter_sub, sub_iterations, sub_final_l1 = pagerank_power_iteration(
        graph=subgraph,
        p=TELEPORT_PROB,
        epsilon=EPSILON,
        max_iter=MAX_ITER,
    )

    print(f"Subgraph iterations      : {sub_iterations}")
    print(f"Subgraph final L1 diff   : {sub_final_l1:.10e}")
    print(f"Subgraph iterative sum   : {pi_iter_sub.sum():.12f}")
    print()

    print("Running PageRank (closed form) on validation subgraph...")
    start_time = time.time()
    pi_closed_sub = pagerank_closed_form(subgraph, p=TELEPORT_PROB)
    closed_runtime = time.time() - start_time

    print(f"Subgraph closed-form sum : {pi_closed_sub.sum():.12f}")
    print(f"Closed-form runtime (s)  : {closed_runtime:.4f}")
    print()

    metrics = compare_vectors(pi_iter_sub, pi_closed_sub)

    print("Comparison Metrics (Validation Subgraph)")
    print("----------------------------------------")
    print(f"L1 difference           : {metrics['L1_difference']:.10e}")
    print(f"L2 difference           : {metrics['L2_difference']:.10e}")
    print(f"Max absolute difference : {metrics['max_absolute_difference']:.10e}")
    print(f"Sum (iterative)         : {metrics['sum_iterative']:.12f}")
    print(f"Sum (closed form)       : {metrics['sum_closed_form']:.12f}")
    print()

    top_iter_sub = top_k_pagerank(pi_iter_sub, sub_n2o, k=min(TOP_K, sub_n))
    top_closed_sub = top_k_pagerank(pi_closed_sub, sub_n2o, k=min(TOP_K, sub_n))

    print_top_k_table("Top Pages by PageRank (Subgraph - Iterative)", top_iter_sub)
    print_top_k_table("Top Pages by PageRank (Subgraph - Closed Form)", top_closed_sub)

    print("=" * 70)
    print("Experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
