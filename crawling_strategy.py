# ============================================================
# AI-Driven Web Crawling Strategy Using PageRank
# Experimental Evaluation on web-Google_10k.txt
#
# This program:
# 1. Loads a directed web graph from an edge-list dataset
# 2. Computes PageRank on the full graph
# 3. Extracts a small high-authority subgraph for illustration
# 4. Constructs a crawlability map for constrained selection
# 5. Compares:
#       - Baseline prioritization: PageRank only
#       - Enhanced prioritization: crawlability-constrained authority
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Set
import random
import networkx as nx


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DATASET_PATH = "web-Google_10k.txt"

# Number of pages to show in the small experimental graph
SMALL_GRAPH_SIZE = 15

# Crawl budget for prioritization
K = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Probability that a page is crawlable
CRAWLABLE_PROBABILITY = 0.7

# Minimum number of blocked pages required inside the baseline top-k
# This helps ensure the baseline and enhanced results differ.
MIN_BLOCKED_IN_TOP_K = 1

# Maximum retries when generating a crawlability map
MAX_CRAWLABILITY_RETRIES = 200


# ------------------------------------------------------------
# 1. Load full directed graph from edge-list dataset
# ------------------------------------------------------------
def load_directed_graph(filepath: str) -> nx.DiGraph:
    """
    Load a directed graph from an edge-list file.

    Each valid non-comment line is expected to contain:
        source_node target_node

    The original dataset uses numeric node IDs. For readability,
    each node is relabeled using the form:
        page_12345

    Parameters
    ----------
    filepath : str
        Path to the web graph edge-list file.

    Returns
    -------
    nx.DiGraph
        Directed graph containing all nodes and edges from the file.
    """
    G = nx.DiGraph()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            source, target = line.split()
            source_label = f"page_{source}"
            target_label = f"page_{target}"

            G.add_edge(source_label, target_label)

    return G


# ------------------------------------------------------------
# 2. Convert full graph to assignment-style dictionary
# ------------------------------------------------------------
def graph_to_dict(G: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Convert a directed NetworkX graph into the assignment input format:

        graph[url] = list of outlinks

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph.

    Returns
    -------
    dict
        Mapping from node/page label to list of outgoing links.
    """
    graph_dict = {node: list(G.successors(node)) for node in G.nodes()}

    # Ensure every node appears as a key, even if it has no outlinks
    for node in G.nodes():
        graph_dict.setdefault(node, [])

    return graph_dict


# ------------------------------------------------------------
# 3. Compute PageRank on full graph
# ------------------------------------------------------------
def compute_pagerank(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute PageRank scores on the full directed graph.

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    dict
        Mapping from node/page label to PageRank score.
    """
    return nx.pagerank(G, alpha=0.85)


# ------------------------------------------------------------
# 4. Extract a small high-authority experimental graph
# ------------------------------------------------------------
def extract_small_high_authority_graph(
    G: nx.DiGraph,
    pagerank: Dict[str, float],
    small_size: int = 15
) -> Tuple[Dict[str, List[str]], Dict[str, float], List[str]]:
    """
    Build a small experimental graph for the crawl-prioritization study.

    Strategy:
    - Start from high-PageRank nodes on the full graph
    - Add neighboring nodes to preserve local structure
    - Retain a small set of selected nodes
    - Return the induced subgraph as a dictionary

    This produces a compact graph that preserves meaningful
    authority patterns from the larger dataset.

    Parameters
    ----------
    G : nx.DiGraph
        Full graph.
    pagerank : dict
        Full-graph PageRank scores.
    small_size : int
        Number of nodes to keep in the small experimental graph.

    Returns
    -------
    small_graph : dict
        Assignment-style adjacency dictionary for the small graph.
    small_pagerank : dict
        PageRank values for nodes in the small graph, using the
        full-graph PageRank scores.
    selected_nodes_sorted : list
        Selected nodes sorted by PageRank descending.
    """
    ranked_nodes = sorted(pagerank, key=pagerank.get, reverse=True)

    seed_count = max(5, small_size // 2)
    seed_nodes = ranked_nodes[:seed_count]

    selected: Set[str] = set(seed_nodes)

    for node in seed_nodes:
        for nbr in G.successors(node):
            selected.add(nbr)
            if len(selected) >= small_size:
                break
        if len(selected) >= small_size:
            break

        for nbr in G.predecessors(node):
            selected.add(nbr)
            if len(selected) >= small_size:
                break
        if len(selected) >= small_size:
            break

    if len(selected) < small_size:
        for node in ranked_nodes:
            selected.add(node)
            if len(selected) >= small_size:
                break

    selected_nodes_sorted = sorted(selected, key=lambda x: pagerank[x], reverse=True)
    selected_nodes_sorted = selected_nodes_sorted[:small_size]
    selected_set = set(selected_nodes_sorted)

    small_graph = {
        node: [nbr for nbr in G.successors(node) if nbr in selected_set]
        for node in selected_nodes_sorted
    }

    for node in selected_nodes_sorted:
        small_graph.setdefault(node, [])

    small_pagerank = {node: pagerank[node] for node in selected_nodes_sorted}

    return small_graph, small_pagerank, selected_nodes_sorted


# ------------------------------------------------------------
# 5. Create a random but meaningful crawlability map
# ------------------------------------------------------------
def build_meaningful_crawlability(
    selected_nodes_sorted: List[str],
    k: int,
    crawlable_probability: float = 0.7,
    min_blocked_in_top_k: int = 1,
    max_retries: int = 200,
    random_seed: int = 42
) -> Dict[str, bool]:
    """
    Create a crawlability dictionary while ensuring that the
    comparison remains informative.

    Design:
    - Assign crawlability probabilistically
    - Require at least `min_blocked_in_top_k` blocked pages among the
      baseline top-k nodes so the constrained method differs from the baseline
    - Require at least k crawlable pages overall so the constrained method
      can still return k results

    Parameters
    ----------
    selected_nodes_sorted : list
        Selected nodes sorted by PageRank descending.
    k : int
        Crawl budget.
    crawlable_probability : float
        Probability that any given page is crawlable.
    min_blocked_in_top_k : int
        Minimum number of blocked pages required within the top-k nodes.
    max_retries : int
        Maximum number of attempts to generate a valid crawlability map.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        crawlable[node] = True/False
    """
    rng = random.Random(random_seed)
    top_k_nodes = selected_nodes_sorted[:k]

    for _ in range(max_retries):
        crawlable = {
            node: (rng.random() < crawlable_probability)
            for node in selected_nodes_sorted
        }

        num_blocked_in_top_k = sum(
            1 for node in top_k_nodes if not crawlable[node]
        )
        num_crawlable_total = sum(crawlable.values())

        if num_blocked_in_top_k >= min_blocked_in_top_k and num_crawlable_total >= k:
            return crawlable

    # Fallback: enforce a valid comparison setup if repeated sampling fails
    crawlable = {node: True for node in selected_nodes_sorted}
    if selected_nodes_sorted:
        crawlable[selected_nodes_sorted[0]] = False
    return crawlable


# ------------------------------------------------------------
# 6. Baseline prioritization
# ------------------------------------------------------------
def get_top_k_urls(pagerank: Dict[str, float], k: int) -> List[str]:
    """
    Baseline method:
    Return top-k pages ranked only by PageRank.
    """
    return sorted(pagerank, key=pagerank.get, reverse=True)[:k]


# ------------------------------------------------------------
# 7. Enhanced prioritization
# ------------------------------------------------------------
def get_top_k_crawlable_urls(
    graph: Dict[str, List[str]],
    pagerank: Dict[str, float],
    crawlable: Dict[str, bool],
    k: int,
    min_out_degree: int = 0
) -> List[str]:
    """
    Enhanced method:
    Return top-k pages that are crawlable and high-authority.

    Note:
    The out-degree filter is not applied in this implementation.
    The enhanced strategy differs from the baseline only through the
    crawlability constraint.

    Parameters
    ----------
    graph : dict
        Web graph in dictionary form. Included to preserve function structure.
    pagerank : dict
        Precomputed PageRank values.
    crawlable : dict
        Crawlability indicator.
    k : int
        Number of pages to return.
    min_out_degree : int
        Ignored in this version.

    Returns
    -------
    list
        Top-k crawlable pages ranked by PageRank.
    """
    valid_urls = [url for url in pagerank if crawlable.get(url, False)]
    return sorted(valid_urls, key=lambda x: pagerank[x], reverse=True)[:k]


# ------------------------------------------------------------
# 8. Utility functions
# ------------------------------------------------------------
def compute_out_degree(graph: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Compute out-degree for each page in the small graph.
    """
    return {url: len(outlinks) for url, outlinks in graph.items()}


def print_ranked_table(
    urls: List[str],
    pagerank: Dict[str, float],
    crawlable: Dict[str, bool],
    out_degree: Dict[str, int],
    title: str
) -> None:
    """
    Print a formatted ranking table.
    """
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)
    print(f"{'Rank':<6}{'URL':<22}{'PageRank':<15}{'Crawlable':<12}{'Out-Degree':<12}")
    print("-" * 95)

    for i, url in enumerate(urls, start=1):
        print(
            f"{i:<6}"
            f"{url:<22}"
            f"{pagerank[url]:<15.8f}"
            f"{str(crawlable.get(url, False)):<12}"
            f"{out_degree.get(url, 0):<12}"
        )


def print_dictionary_preview(
    graph: Dict[str, List[str]],
    pagerank: Dict[str, float],
    crawlable: Dict[str, bool]
) -> None:
    """
    Print the small graph, PageRank, and crawlability dictionaries.
    """
    print("\n" + "=" * 95)
    print("Small Directed Web Graph (Assignment-Style Dictionary)")
    print("=" * 95)
    print("graph = {")
    for node, outlinks in graph.items():
        print(f'    "{node}": {outlinks},')
    print("}")

    print("\n" + "=" * 95)
    print("Precomputed PageRank Scores (Assignment-Style Dictionary)")
    print("=" * 95)
    print("pagerank = {")
    for node, score in pagerank.items():
        print(f'    "{node}": {score:.8f},')
    print("}")

    print("\n" + "=" * 95)
    print("Crawlability Map")
    print("=" * 95)
    print("crawlable = {")
    for node, allowed in crawlable.items():
        print(f'    "{node}": {allowed},')
    print("}")


def compare_baseline_vs_enhanced(
    baseline_urls: List[str],
    enhanced_urls: List[str]
) -> None:
    """
    Compare the baseline and enhanced selections.
    """
    print("\n" + "=" * 95)
    print("Comparison of Results")
    print("=" * 95)

    print("\nBaseline Top-k URLs:")
    for i, url in enumerate(baseline_urls, start=1):
        print(f"{i}. {url}")

    print("\nEnhanced Top-k URLs:")
    for i, url in enumerate(enhanced_urls, start=1):
        print(f"{i}. {url}")

    excluded = [url for url in baseline_urls if url not in enhanced_urls]
    added = [url for url in enhanced_urls if url not in baseline_urls]

    print("\nURLs selected by baseline but excluded by enhanced method:")
    if excluded:
        for url in excluded:
            print(f"- {url}")
    else:
        print("- None")

    print("\nURLs newly selected by enhanced method:")
    if added:
        for url in added:
            print(f"- {url}")
    else:
        print("- None")


# ------------------------------------------------------------
# 9. Main experiment
# ------------------------------------------------------------
def main() -> None:
    dataset_file = Path(DATASET_PATH)

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_file.resolve()}\n"
            f"Make sure '{DATASET_PATH}' is in the same folder as this script."
        )

    print("=" * 95)
    print("Loading real web graph dataset")
    print("=" * 95)

    G = load_directed_graph(str(dataset_file))
    full_pagerank = compute_pagerank(G)

    print(f"Dataset file            : {dataset_file.name}")
    print(f"Number of nodes         : {G.number_of_nodes()}")
    print(f"Number of edges         : {G.number_of_edges()}")

    print("\nTop 10 pages by PageRank on full graph:")
    top_10_full = sorted(full_pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (node, score) in enumerate(top_10_full, start=1):
        print(f"{rank:>2}. {node:<20} {score:.8f}")

    small_graph, small_pagerank, selected_nodes_sorted = extract_small_high_authority_graph(
        G=G,
        pagerank=full_pagerank,
        small_size=SMALL_GRAPH_SIZE
    )

    crawlable = build_meaningful_crawlability(
        selected_nodes_sorted=selected_nodes_sorted,
        k=K,
        crawlable_probability=CRAWLABLE_PROBABILITY,
        min_blocked_in_top_k=MIN_BLOCKED_IN_TOP_K,
        max_retries=MAX_CRAWLABILITY_RETRIES,
        random_seed=RANDOM_SEED
    )

    out_degree = compute_out_degree(small_graph)

    baseline_top_k = get_top_k_urls(small_pagerank, K)

    enhanced_top_k = get_top_k_crawlable_urls(
        graph=small_graph,
        pagerank=small_pagerank,
        crawlable=crawlable,
        k=K,
        min_out_degree=0
    )

    print("\n" + "=" * 95)
    print("AI-Driven Web Crawling Experiment")
    print("=" * 95)
    print(f"Small graph size        : {len(small_graph)}")
    print(f"Crawl budget k          : {K}")
    print(f"Random seed             : {RANDOM_SEED}")
    print(f"Crawlable probability   : {CRAWLABLE_PROBABILITY}")
    print(f"Min blocked in top-k    : {MIN_BLOCKED_IN_TOP_K}")

    print_dictionary_preview(
        graph=small_graph,
        pagerank=small_pagerank,
        crawlable=crawlable
    )

    print_ranked_table(
        baseline_top_k,
        small_pagerank,
        crawlable,
        out_degree,
        "Baseline Prioritization (PageRank Only)"
    )

    print_ranked_table(
        enhanced_top_k,
        small_pagerank,
        crawlable,
        out_degree,
        "Enhanced Prioritization (Crawlability-Constrained Authority)"
    )

    compare_baseline_vs_enhanced(baseline_top_k, enhanced_top_k)

    print("\n" + "=" * 95)
    print("Interpretation")
    print("=" * 95)
    print(
        "The baseline method ranks pages solely by PageRank and may therefore "
        "select high-authority pages that cannot be retrieved. The enhanced method "
        "applies the crawlability constraint before ranking, ensuring that only "
        "feasible pages are selected. In this experiment, crawlability is assigned "
        "probabilistically, with safeguards that ensure at least one high-authority "
        "page in the baseline top-k is blocked. This produces a clear comparison "
        "between unconstrained and feasibility-aware prioritization."
    )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()  