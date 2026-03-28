# PageRank and AI-Driven Web Crawling Strategy

This repository contains the implementation and evaluation of:

1. **PageRank computation and validation**
2. **AI-driven web crawling prioritization strategy**

Both components are designed to support large-scale web graph analysis and demonstrate how PageRank can be applied in practical systems such as web crawling for AI data collection.

---

## 📁 Files

### 1. PageRank Implementation
- **File:** `pagerank_assignment.py`  
- **Reference:** :contentReference[oaicite:0]{index=0}  

Implements:
- PageRank using **power iteration**
- PageRank using **closed-form solution**
- Validation between both methods
- Distribution analysis on large graphs

---

### 2. crawling_strategy
- **File:** `crawler_submission_ready.py`  
- **Reference:** :contentReference[oaicite:1]{index=1}  

Implements:
- PageRank-based crawl prioritization
- Crawlability-constrained selection
- Experimental comparison between strategies

### 3. web-Google_10k Dataset
- **File:** `web-Google_10k.txt`

Contains:
- A real-world directed web graph (edge list format)
- ~10,000 nodes (webpages)
- ~78,000 edges (hyperlinks between pages)

Used for:
- Large-scale PageRank computation
- Empirical analysis of PageRank distribution
- Subgraph extraction for validation and experiments
- Evaluation of crawling prioritization strategies
---

## 🚀 Overview

### Objective

This project addresses two key problems:

1. **Compute PageRank efficiently on large web graphs**
2. **Use PageRank to design an effective web crawling strategy for AI systems**

---

## 📊 Part 1: PageRank Computation and Validation

### Key Features

- **Power Iteration Method**
  - Scalable to large graphs (10,000+ nodes)
  - Iterative convergence with L1 error threshold

- **Closed-Form Solution**
  - Based on linear system:
    ```
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
  - Cumulative distribution (CDF)

### Key Insight

PageRank distribution is **highly skewed**, where:
- A small number of nodes hold very high importance
- Most nodes have very low PageRank

This reflects real-world web structures with **hub nodes and authority concentration**.

---

## 🧠 Part 2: AI-Driven Crawling Strategy

### Problem

Given:
- A web graph
- Precomputed PageRank scores

Select the **top-k URLs to crawl** under real-world constraints.

---

### Baseline Strategy

**PageRank-only prioritization**

- Rank pages by PageRank
- Select top-k pages

**Limitation:**
- May select pages that **cannot be crawled** (e.g., blocked by robots.txt)

---

### Enhanced Strategy

**Crawlability-Constrained Authority**

A page is selected only if:

- It has **high PageRank**
- It is **crawlable**

Formally:Select top-k pages such that:

High PageRank
crawlable(i) = True


---

### Experimental Design

- Dataset: `web-Google_10k.txt`
- Extract small high-authority subgraph
- Simulate crawlability probabilistically
- Compare:
  - Baseline vs Enhanced selection

---

### Key Findings

- Enhanced strategy:
  - Avoids non-crawlable pages
  - Maintains high authority quality
  - Improves resource efficiency

- Baseline strategy:
  - Can waste crawl budget on inaccessible pages

---

## ⚙️ How to Run

### Requirements

```bash
pip install networkx numpy matplotlib
