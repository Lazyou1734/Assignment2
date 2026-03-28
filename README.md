readme_md = """
# PageRank and AI-Driven Web Crawling Strategy

This repository contains the implementation and evaluation of:

1. **PageRank computation and validation**
2. **AI-driven web crawling prioritization strategy**

Both components are designed to support large-scale web graph analysis and demonstrate how PageRank can be applied in practical systems such as web crawling for AI data collection.

---

## Files

### 1. PageRank Implementation
- **File:** `pagerank_assignment.py`

Implements:
- PageRank using **power iteration**
- PageRank using a **closed-form solution**
- Validation between both methods
- Distribution analysis on large graphs

---

### 2. Crawling Strategy
- **File:** `crawling_strategy.py`

Implements:
- PageRank-based crawl prioritization
- Crawlability-constrained selection
- Experimental comparison between strategies

---

### 3. web-Google_10k Dataset
- **File:** `web-Google_10k.txt`

Contains:
- A real-world directed web graph in **edge list format**
- Approximately **10,000 nodes** (webpages)
- Approximately **78,000 edges** (hyperlinks between pages)

Used for:
- Large-scale PageRank computation
- Empirical analysis of PageRank distribution
- Subgraph extraction for validation and experiments
- Evaluation of crawling prioritization strategies

---

## Overview

### Objective

This project addresses two key problems:

1. **Computing PageRank efficiently on large web graphs**
2. **Using PageRank to design an effective web crawling strategy for AI systems**

---

## Part 1: PageRank Computation and Validation

### Key Features

- **Power Iteration Method**
  - Scales to large graphs with 10,000+ nodes
  - Uses iterative convergence with an L1 error threshold

- **Closed-Form Solution**
  - Based on the linear system:
    ```text
    π = (p/n) (I - (1-p)P)^(-1) 1
    ```
  - Used for validation on smaller subgraphs

- **Validation Metrics**
  - L1 difference
  - L2 difference
  - Maximum absolute difference

- **Distribution Analysis**
  - Histogram of PageRank scores
  - Log-scaled distribution
  - Cumulative distribution function (CDF)

### Key Insight

The PageRank distribution is **highly skewed**:
- A small number of nodes have very high importance
- Most nodes have very low PageRank scores

This reflects real-world web structures, where **hub nodes** and **authoritative pages** concentrate attention and influence.

---

## Part 2: AI-Driven Crawling Strategy

### Problem

Given:
- A web graph
- Precomputed PageRank scores

The goal is to select the **top-k URLs to crawl** under realistic constraints.

---

### Baseline Strategy

**PageRank-only prioritization**

- Rank pages by PageRank
- Select the top-k pages

**Limitation:**
- This may select pages that **cannot be crawled** (for example, pages blocked by `robots.txt`)

---

### Enhanced Strategy

**Crawlability-Constrained Authority**

A page is selected only if it satisfies both conditions:
- It has **high PageRank**
- It is **crawlable**

Formally, the strategy selects the top-k pages among those that satisfy:

```text
crawlable(i) = True
```

while preserving the highest available PageRank scores.

---

### Experimental Design

- Dataset: `web-Google_10k.txt`
- Extract a small high-authority subgraph
- Simulate crawlability probabilistically
- Compare:
  - Baseline selection
  - Enhanced selection

---

### Key Findings

- **Enhanced strategy**
  - Avoids non-crawlable pages
  - Maintains high-authority selections
  - Improves crawl budget efficiency

- **Baseline strategy**
  - Can waste crawl budget on inaccessible pages

---

## How to Run

### Requirements

```bash
pip install networkx numpy matplotlib
```

### Example

```bash
python pagerank_assignment.py
python crawling_strategy.py
```

---

## Summary

This project shows that:

- PageRank can be computed efficiently on large web graphs
- The closed-form solution is useful for validating iterative results on smaller subgraphs
- Real-world PageRank distributions are strongly skewed
- Crawl prioritization improves when structural importance is combined with crawlability constraints

This makes PageRank a useful foundation for both **web graph analysis** and **practical AI-oriented crawling strategy design**.
"""