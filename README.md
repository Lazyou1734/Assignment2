# AI-Driven Web Crawling Strategy Using PageRank

## Overview
This project implements and evaluates a **PageRank-based crawl prioritization strategy** for AI-driven web crawling systems. It simulates how a crawler can prioritize newly discovered web pages under a limited crawl budget.

Two strategies are compared:

- **Baseline**: prioritization using PageRank only
- **Enhanced**: prioritization using PageRank with crawlability constraints

The objective is to show how incorporating feasibility constraints, such as crawl permissions, can improve the effectiveness of crawl resource allocation.

## Key Features

- Computes **PageRank on a real-world web graph dataset** (`web-Google_10k.txt`)
- Extracts a **high-authority subgraph** for controlled experimentation
- Simulates **crawlability constraints**
- Implements and compares:
  - **Baseline prioritization** using PageRank only
  - **Enhanced prioritization** using crawlability-constrained authority
- Provides clear **ranking outputs and comparison analysis**
- Includes **visualization of PageRank distribution**

## Project Structure

```text
.
├── pagerank_submission_ready.py
├── crawler_submission_ready.py
├── web-Google_10k.txt
└── README.md