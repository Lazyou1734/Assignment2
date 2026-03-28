AI-Driven Web Crawling Strategy Using PageRank
Overview

This project implements and evaluates a PageRank-based crawl prioritization strategy for AI-driven web crawling systems. It simulates how a crawler (e.g., GPTBot-like systems) can prioritize newly discovered web pages under a limited crawl budget.

Two strategies are compared:

Baseline: Prioritization using PageRank only
Enhanced: Prioritization using PageRank with crawlability constraints

The goal is to demonstrate how incorporating feasibility constraints (e.g., robots.txt restrictions) improves the effectiveness of crawl resource allocation.

Key Features
Computes PageRank on a real-world web graph dataset (web-Google_10k.txt)
Extracts a high-authority subgraph for controlled experimentation
Simulates crawlability constraints
Implements and compares:
PageRank-only prioritization
Crawlability-constrained prioritization
Provides clear ranking outputs and comparison analysis
Includes visualization of PageRank distribution
Project Structure
.
├── pagerank_submission_ready.py      # PageRank computation and analysis
├── crawler_submission_ready.py      # Crawl prioritization experiment
├── web-Google_10k.txt               # Dataset (edge list format)
└── README.md
Dataset

The project uses a subset of the Google web graph dataset:

File: web-Google_10k.txt

Format: Each line represents a directed edge

source_node target_node

Nodes are relabeled into a readable format:

page_<node_id>
Installation

Ensure Python 3.8+ is installed.

Install dependencies:

pip install networkx numpy matplotlib
Usage
1. PageRank Computation

Run the PageRank analysis and distribution visualization:

python pagerank_submission_ready.py

This will:

Load the full graph
Compute PageRank using power iteration
Display top-ranked nodes
Generate PageRank distribution plots
2. Crawling Strategy Experiment

Run the crawl prioritization comparison:

python crawler_submission_ready.py

This will:

Extract a small high-authority graph
Generate a crawlability map
Compare:
Baseline (PageRank only)
Enhanced (crawlability-constrained)
Output ranked tables and differences
Methodology
PageRank-Based Prioritization

Pages are ranked based on their PageRank score:

High PageRank → structurally important pages
Typically linked by many authoritative sources
More likely to contain high-quality content
Enhanced Heuristic: Crawlability-Constrained Authority

The enhanced method applies an additional constraint:

Only pages that permit crawling are considered

Formally:

Select top-k pages such that:
1. PageRank is high
2. crawlable(page) = True
Key Insight

While PageRank identifies high-authority pages, it does not account for access constraints.

The enhanced strategy:

Avoids selecting non-crawlable pages
Preserves most high-value pages
Improves effective utilization of crawl budget

This aligns better with real-world crawling systems.

Example Output (Simplified)
Baseline Top-5:
1. page_486980
2. page_285814
3. page_226374
4. page_163075
5. page_555924  (not crawlable)

Enhanced Top-5:
1. page_486980
2. page_285814
3. page_226374
4. page_163075
5. page_32163   (replaces blocked page)
Limitations
Crawlability is simulated probabilistically, not derived from real robots.txt files
Only a single constraint (crawlability) is considered
Experiments use a reduced subgraph for interpretability
Future Improvements
Integrate real robots.txt parsing
Incorporate additional signals:
Content freshness
Duplicate detection
Domain-level relevance
Extend to large-scale distributed crawling systems
Technologies Used
Python
NetworkX (graph processing)
NumPy (numerical operations)
Matplotlib (visualization)
Author

Zhijian Dong