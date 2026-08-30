# Findly: Conversational Shopping Copilot

**TikTok TechJam 2026**  
**Track 4: Shopping Copilot**  
**Team Members: Rann Liu, Aricia Ng**

## 1. Project Overview

**Findly** is a conversational shopping copilot designed to help users discover suitable products through natural, multi-turn conversation.

Traditional product search often expects users to know exactly what they want from the beginning. In reality, shopping is usually iterative and shoppers tend to be indecisive. A shopper may start with a broad request such as:

> “I’m looking for a jacket.”

and only later reveal that they want something waterproof, black, lightweight, and suitable for hiking.

Findly handles this by maintaining a structured understanding of the user's preferences throughout the conversation, asking useful clarification questions, retrieving relevant products from the catalog, and continually refining its recommendations as more information becomes available.

Rather than treating every message as an independent search query, Findly treats shopping as an evolving conversation.

Our final system combines:

- Multi-turn conversational memory
- Structured preference extraction
- Intent-override handling
- Clarification-question selection
- Structured BM25 retrieval using SQLite FTS5
- Cross-turn candidate consensus
- Product category/type-aware reranking
- Debug logging and offline diagnostic tooling

The final agent achieved:

The final agent achieved:

```text
Hit@10: 0.770
MRR: 0.433151
MTTC: 5.215
Efficiency: 0.5785
Recommended Technical Score: 0.630645
```

on the 200-session public evaluation set.

## 2. Solution's Limitations and What We Would Improve Given More Time

Although the problem statement proposes separate buying/browsing routing and dense/LLM retrieval, these were **not included in our final agent**. During development, we tested more complex routing and semantic retrieval approaches, but they did not outperform our deterministic retrieval system on the supplied evaluation set. We therefore retained the architecture that produced the strongest measured performance.

Given more time, we could seek to incorporate a **hybrid semantic retrieval and LLM reranking pipeline** by combining our agent with **vector-based dense retrieval** to retrieve semantically similar products, then using an LLM to rerank only a small shortlist of candidates based on the full conversational context and accumulated user preferences, personalising the results accordingly.

For example, Findly could first use BM25 to efficiently retrieve the top 50 most relevant products. These candidates, together with the user's accumulated preferences and conversation context, could then be passed to an LLM for semantic reranking.

The LLM would evaluate how well each product matches the user's overall intent and reorder the shortlist before returning the final Top 10 recommendations.

This keeps the initial retrieval fast while using the LLM only on a smaller candidate set where deeper contextual understanding is most useful.

## 3. Setup and Installation

### Requirements

Recommended:

```text
Python 3.10+
Git
SQLite with FTS5 support
```

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Tiktok-TechJam-FattyCoders
```

Optional but recommended: create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Dependencies
```markdown

Findly does not require any third-party Python packages. The final agent uses only Python's standard library, including SQLite with FTS5 support, and does not require any external API keys.
```

The final Findly retrieval pipeline itself is intentionally lightweight and does not require an external inference API.

### 3.1 Dataset Setup

Ensure the following challenge files are in the correct locations:

```text
data/
├── catalog.jsonl
└── public_set.jsonl
```

The repository should resemble:

```text
project-root/
├── starter/
│   └── agent.py
├── evaluator/
│   └── local_evaluator.py
├── data/
│   ├── catalog.jsonl
│   └── public_set.jsonl
└── README.md
```

### 3.2 Running Findly

From the repository root, run:

```bash
python -m evaluator.local_evaluator
```

No external API key is required.

The final submission configuration is enabled directly in the agent and does not require special environment variables.

Expected results on the supplied 200-session public evaluation set are approximately:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.77,
  "mrr": 0.433151,
  "mttc": 5.215,
  "efficiency": 0.5785,
  "recommended_technical_score": 0.630645
}
```

## 4. Steps to Reproduce Our Results

1. Clone the repository.
2. Ensure the challenge dataset exists under `data/`.
3. Install any required Python dependencies.
4. Ensure no experimental environment variables are active.
5. From the repository root, run:

```bash
python -m evaluator.local_evaluator
```

6. The evaluator loads all 200 public sessions.
7. Each session is passed to Findly through the official agent interface.
8. The evaluator measures Hit@10, MRR and MTTC.
9. The final expected technical score is approximately:

```text
0.630645
```

Because Findly is deterministic and does not depend on external APIs, repeated runs using the same code and dataset should reproduce the same results.

## 5. Team Member Contributions

**Rann Liu**

Contributed to the development, experimentation, debugging and evaluation of Findly, including the conversational agent architecture, retrieval/ranking pipeline, iterative performance testing and final integration.

**Aricia Ng**

Contributed to the development, testing and refinement of Findly, including conversational behaviour, solution ideation, evaluation, debugging and project preparation.

Both team members collaborated on:

- Defining the shopping-copilot approach
- Iterating on the agent design
- Testing and debugging
- Reviewing evaluation results
- Refining the final solution
- Preparing the final submission
