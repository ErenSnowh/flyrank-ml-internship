# Standout ML Intern Operating Rules — Vinayak (ErenSnowh)

These instructions guide how Antigravity pair-programs with Vinayak in this workspace to ensure top-tier, publication-grade results across every assignment and the final capstone.

## 1. Ground Rules & Workflow
- **Skill-First Approach**: Always check `skills/README.md` and load the specific skill named for the task.
- **Reproducibility**: Fix random seeds (`random_state=42`), log exact shapes, and ensure notebooks run cleanly top-to-bottom without manual intervention.
- **Safe Data Handling**: Never commit dataset CSVs (CI protection). Never print raw queries, URLs, client names, or domain strings.

## 2. Technical Standards to Stand Out
- **Zero Leakage**: Strictly exclude target sources (`trend_pct`, `trend_direction`) and downstream product outputs (`health_score`, `priority_score`). Always include programmatic leakage assertions in notebook cells.
- **Client-Holdout Splits**: Validate models by holding out ~20% of whole clients (`client_id` / `client_hash_id`) to test true cross-client generalization.
- **Top-K Evaluation**: Primary metrics must be **Precision@20** and **Precision@50**, reflecting human editorial review capacity. Always report the base rate alongside Precision@K for clear lift demonstration.
- **Reason Codes & Action Mapping**: Every scored page must output readable reason codes and a concrete action label (`refresh`, `refresh_and_review_ctr`, `expand_and_refresh`, `monitor`).
- **Skeptic's Audit**: Include a "what would make it wrong" critique for top picks to show senior-level engineering judgment.

## 3. Communication & Paper Standards
- **Defensible Language**: Use cautious, scientific language: *observed, directional, decision-support*. Never claim causal impact without an A/B test design. Never claim to have cracked Google's search algorithm.
- **End-to-End Artifact Integrity**: All generated CSVs, JSON receipts, and SVG charts live in `work/outputs/`. All notebooks in `work/notebooks/` must be executed and committed.
