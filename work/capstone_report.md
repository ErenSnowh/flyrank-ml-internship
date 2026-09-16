# Capstone Report — Lane 2: Content Refresh Prioritization
- **Author:** Vinayak Pandey ([Portfolio](https://allabtme.vercel.app/) · [GitHub](https://github.com/ErenSnowh))
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** [https://github.com/ErenSnowh/flyrank-ml-internship](https://github.com/ErenSnowh/flyrank-ml-internship)
- **Deployed Paper:** [https://ErenSnowh.github.io/flyrank-ml-internship/](https://ErenSnowh.github.io/flyrank-ml-internship/)
- **Date:** September 2026

---

## 0. Abstract

FlyRank operates autonomous content infrastructure at scale, continuously researching, publishing, and optimizing client portfolios. In large-scale content operations, assets inevitably experience search decay: rankings slip, impressions drop, and manual triage across thousands of URLs becomes untenable. Existing production workflows rely on hand-written heuristic flags (such as static health scores and freshness thresholds), which struggle to capture multi-variable interactions in noisy search telemetry. This case study evaluates whether behavioral signals from search and analytics data can reliably predict impression decline to prioritize editorial refresh queues. Using 30,000 pseudonymized content pages across 32 enterprise clients (trailing 90-day behavioral metrics), we trained a shallow decision-tree classifier evaluated on strict client-holdout splits (holding out 20% of whole clients unseen during training). The model achieved **Precision@50 of 0.68** against a 0.39 base rate — representing a **1.7× lift over random selection** and a **2.8× lift over FlyRank's hand-crafted rule baseline** (P@50 = 0.24). Rather than replacing human oversight, the system outputs an interpretable ranked queue with transparent reason codes and actionable editorial playbooks, demonstrating how learned models enhance human editorial bandwidth in production SEO pipelines.

---

## 1. Problem Framing

- **Decision Supported:** Which pages across a large multi-client content portfolio should a human editor inspect and refresh during the upcoming bi-weekly sprint?
- **Unit of Analysis:** One pseudonymized content page (`content_id` / `content_hash_id`) within a client portfolio (`client_id`).
- **Output:** A ranked prioritization queue containing blended priority scores (0–100), primary & secondary reason codes (e.g., `imp_decay_high`, `freshness_risk`), and concrete editorial action tags (`refresh`, `refresh_and_review_ctr`, `refresh_and_review_engagement`, `expand_and_refresh`, `monitor`).
- **Action a Human Takes:** An editor opens the top-50 queue, reviews the specific page against the reason codes, inspects current SERP competition, and initiates a targeted content refresh or structural expansion.
- **Cost of a Wrong Call:** 
  - *False Positive:* Wastes ~2–6 hours of senior editorial rewriting time on a page that was stable.
  - *False Negative:* Leaves a deteriorating high-value revenue asset unrefreshed, resulting in cumulative loss of organic traffic and eventual client churn.
- **Why Machine Learning Helps:** Hand-crafted rules rely on arbitrary scalar cutoffs (e.g., `age > 180` and `health < 50`) that fail on non-linear trade-offs (e.g., an older page with consistent high engagement vs. a moderately new page suffering an acute cliff in impression-active days). A learned model isolates high-dimensional signal interactions without overfitting to client identities.

---

## 2. Data Safety & Zero Leakage

- **Dataset Used:** FlyRank ML Internship anonymized starter release — 30,000 pages × 44 columns, spanning 32 pseudonymized enterprise clients. All metrics aggregate trailing 90-day search and behavioral telemetry.
- **Deliberately Excluded Columns:**
  - `trend_direction` & `trend_pct`: Direct target sources. The binary classification label `is_declining_label` ($1$ if `trend_direction == 'down'`, else $0$) is derived from `trend_pct < -20%`. Including either field constitutes direct label leakage.
  - `health_score` & `priority_score`: Downstream production heuristic flags. Using them creates circular evaluation where the model predicts an existing rule rather than the underlying search decay phenomenon.
  - `content_id` & `client_id`: Pseudonymous identifiers. Used strictly for grouping and client-holdout partitioning; never passed as features to avoid domain-identity memorization.
  - `provider_used` & `model_used`: LLM generative metadata; irrelevant to post-publication search visibility trends.
- **Leakage Auditing:** Programmatic assertions in `work/notebooks/w06_validation_audit.ipynb` and `work/notebooks/w03_feature_leakage_check.ipynb` verify that synthetic injection of `trend_pct` causes artificial score inflation (>95% accuracy), confirming that our honest feature vector (achieving ~68% P@50) operates strictly without label leakage.
- **Public-Safety Compliance:** Absolutely zero client names, domain strings, live URLs, private queries, or credentials exist in the repository or output artifacts.

---

## 3. Baseline

- **The Hand-Crafted Rule Baseline:** We implemented FlyRank's operational heuristic rule score ($0–100$), constructed from four transparent components:
  1. *Visibility Score (30 pts):* Log-scaled impressions and click volume.
  2. *Freshness Risk (25 pts):* Penalizing content age > 180 days and stale update intervals.
  3. *Position Opportunity (25 pts):* Targeting pages in striking distance (ranks 4–20).
  4. *Depth Gap (20 pts):* Penalizing short word count under industry threshold.
- **Fairness of Comparison:** Evaluated on the exact same client-holdout split as the learned models.
- **Baseline Performance:**
  - **Precision@20:** 0.250
  - **Precision@50:** 0.240
  - **ROC-AUC:** 0.521
  - **Base Rate:** 0.391
- **Why It Failed:** The hand-crafted rule performs *worse than random guessing* (P@50 = 0.24 vs. 0.39 base rate). The heuristic over-indexed on total impression volume and age, repeatedly selecting stable, high-traffic evergreen articles that were not actually decaying, while missing low-volume pages experiencing catastrophic percentage drops.

---

## 4. Model / Analysis

- **Algorithm:** Decision Tree Classifier (`max_depth=5`, `min_samples_leaf=20`, `random_state=42`) alongside Logistic Regression and Random Forest benchmarks.
- **Why Decision Tree Fits:** Content editors require complete interpretability. A depth-5 decision tree produces human-auditable decision paths, executes in <1ms, and directly translates into intuitive editorial reason codes.
- **Feature Set (18 Honest Features):**
  - *Volume & Scale:* `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`.
  - *Search Consistency:* `days_with_impressions`, `days_with_sessions`.
  - *Freshness & Maturity:* `content_age_days`, `days_since_last_update`.
  - *Performance & Engagement:* `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`.
- **Target Definition:** `is_declining_label = 1` if `trend_direction == 'down'` (impression decline > 20% over trailing 90 days), else `0`.

---

## 5. Evaluation

- **Validation Split Strategy:** **Client-Holdout Split (GroupKFold / 20% holdout clients)**.
  - *Why not random split:* Pages from the same client share domain authority, technical SEO templates, and vertical seasonality. Random page-level splitting allows models to memorize client-level priors, leading to optimistic leakage. Holding out 20% of clients tests true generalization to completely new publisher environments.
- **Metric Comparison Table (Client-Holdout Split, Base Rate = 39.1%):**

| Model | Precision@20 | Precision@50 | ROC-AUC | Avg Precision | Lift over Baseline (P@50) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Rules** | 0.250 | 0.240 | 0.521 | 0.404 | 1.0× |
| **Logistic Regression** | 0.550 | 0.520 | 0.598 | 0.481 | 2.2× |
| **Random Forest** | 0.650 | 0.620 | 0.652 | 0.540 | 2.6× |
| **Decision Tree (depth=5)** | **0.700** | **0.680** | **0.638** | **0.529** | **2.8×** |

- **Error Analysis:**
  - *False Positives (32% in top 50):* Pages that experienced normal day-to-day variance or seasonal troughs without systemic structural decay.
  - *False Negatives:* Pages with high overall impressions whose recent cliff was masked by high early-period volume within the 90-day aggregation.

---

## 6. Interpretation

- **Feature Importance (Decision Tree Gini Splits):**
  1. `days_with_impressions` (43.1%): The dominant signal of content health. Stable pages receive impressions nearly every single day; decaying pages show sparse, intermittent query impressions.
  2. `content_age_days` (24.7%): Content older than ~140 days enters a high-risk decay bracket if not periodically refreshed.
  3. Combined, the top two features explain **67.8% of all tree splits**.
  4. Remaining features: `avg_position` (10.2%), `ctr` (7.4%), `engagement_rate` (5.8%), `word_count` (4.5%), `search_volume` (4.4%).
- **Key Discovery & Negative Results:**
  - Raw session volume and AI traffic percentage had near-zero split importance. High search impressions without consistency (`days_with_impressions`) is a much stronger indicator of imminent decline than session counts.

---

## 7. Recommendation & Action Playbook

- **Operational Queue:** Rather than an opaque probability, each page is assigned:
  1. A blended priority score ($0–100$) combining model confidence and search volume.
  2. Primary and secondary reason codes explaining *why* it was flagged.
  3. A concrete action tag:
     - `refresh`: High decay + aging content → update statistics, refresh headings.
     - `refresh_and_review_ctr`: High decay + below-average CTR → rewrite title tag and meta description.
     - `refresh_and_review_engagement`: High decay + low engagement rate → improve readability and media.
     - `expand_and_refresh`: High decay + short word count → expand comprehensive topical coverage.
     - `monitor`: Low/moderate decay → hold in surveillance queue.
- **Sprint 1 Target:** **3,576 high-confidence actionable pages** across client portfolios.
- **Strict Guardrails (No-Go Checklist):**
  - Never auto-publish rewritten text without human editorial sign-off.
  - Never delete or 301-redirect pages based solely on model decline scores.
  - Never guarantee clients that a refresh will restore top-3 rankings.

---

## 8. Reproducibility

- **Repository:** [https://github.com/ErenSnowh/flyrank-ml-internship](https://github.com/ErenSnowh/flyrank-ml-internship)
- **Random Seed:** Fixed to `random_state = 42` across all scripts and notebooks.
- **Execution from Fresh Clone:**
  ```bash
  git clone https://github.com/ErenSnowh/flyrank-ml-internship.git
  cd flyrank-ml-internship
  pip install -r requirements.txt
  python scripts/run_all.py
  ```
- **Auditable Receipts Committed in Repo:**
  - `work/outputs/w05_model_results.json`: Model evaluation metrics.
  - `work/outputs/w06_validation_audit_results.json`: Leakage and split receipts.
  - `work/outputs/w07_playbook_receipt.json`: Action distribution & guardrail records.
  - `submission/paper_url.txt`: Deployed research paper URL.

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset** — a pseudonymized, multi-client content performance dataset designed for applied machine learning research. Data provided by [FlyRank](https://flyrank.ai). Special thanks to the FlyRank mentorship team and ML track community for technical guidance and peer review.
