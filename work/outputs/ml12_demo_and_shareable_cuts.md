# ML-12: Tell the Story — Showcase Demo Outline & Shareable Cuts

**Author:** Vinayak Pandey  
**Track:** Machine Learning Internship · Lane 2: Content Refresh Prioritization  
**Deployed Research Paper:** [https://ErenSnowh.github.io/flyrank-ml-internship/](https://ErenSnowh.github.io/flyrank-ml-internship/)  
**GitHub Repository:** [https://github.com/ErenSnowh/flyrank-ml-internship](https://github.com/ErenSnowh/flyrank-ml-internship)  
**Personal Portfolio:** [https://allabtme.vercel.app/](https://allabtme.vercel.app/)  

---

## 1. 5-Minute Demo Outline (Week-8 Capstone Showcase)

A structured, timed walk-through designed for the 5-minute showcase. Covering all 5 core pillars:
**Question · Method · One Key Chart · One Honest Result · One Concrete Recommendation**.

### [0:00 – 0:45] Pillar 1: The Question & The FlyRank Content Problem
- **The Hook:** "FlyRank builds content as infrastructure across dozens of enterprise client portfolios. But web content decays silently — rankings slip, impressions drop, and editorial teams can only realistically review about 50 pages per sprint out of thousands."
- **The Core Question:** *Can trailing behavioral signals (impressions, clicks, CTR, engagement, content age, position) predict which content pages are experiencing declining impression trends — well enough to prioritize human editorial review?*
- **Operational Stakes:** False positives waste 2–6 hours of editor time each; false negatives leave high-value decaying pages unrefreshed until client retention is impacted.

### [0:45 – 1:45] Pillar 2: The Method & Honest Validation
- **The Dataset:** 30,000 pseudonymized pages across 32 clients from FlyRank's trailing 90-day search and analytics warehouse. Completely public-safe: no client names, URLs, domains, or raw queries.
- **Zero Leakage Discipline:** Target sources (`trend_pct`, `trend_direction`) and downstream production flags (`health_score`, `priority_score`) were strictly excluded from feature sets and verified via programmatic injection tests.
- **Client-Holdout Splits:** Instead of a naive random train/test split (which leaks client-level domain authority and topical style), we held out 20% of whole clients. The model is evaluated strictly on client ecosystems it has never seen.
- **Models Compared:** Hand-crafted rule baseline (FlyRank's 4-component heuristic) vs. Logistic Regression, Decision Tree ($depth=5$), and Random Forest.

### [1:45 – 2:45] Pillar 3: One Key Chart — Precision@50 Model Comparison
*(Show Figure: Precision@50 across models vs. Baseline on Client Holdout Split)*
- **What this chart shows:** Top-50 precision on the unseen client holdout split across all candidates.
- **The numbers:**
  - **Random Base Rate:** 39.1% (picking pages at random yields ~20 declining pages).
  - **Hand-Crafted Rule Baseline:** 24.0% Precision@50 (only 12 of 50 picks declining — *worse than random*).
  - **Decision Tree ($depth=5$):** **68.0% Precision@50** (34 of 50 picks genuinely declining).
- **The Visual Takeaway:** A simple, interpretable decision tree achieves a **2.8× lift over the rule baseline** and a **1.7× lift over random base rate**.

### [2:45 – 3:45] Pillar 4: One Honest Result & The Skeptic's Audit
- **The Primary Result:** Precision@50 of 0.68 demonstrates that behavioral telemetry carries strong directional signal for triage.
- **Top 2 Feature Drivers:** `days_with_impressions` (43.1% Gini importance) and `content_age_days` (24.7%) account for ~68% of all model splits. Declining pages consistently show fragmented search visibility combined with maturing age.
- **The Skeptic's Audit (Honest Limitations):**
  1. *32% False Positive Rate in Top 50:* 16 of every 50 flagged pages are not declining. This is why the system must remain decision-support, never autonomous publishing.
  2. *Observational, not Causal:* We observe association with historical decline; we cannot claim that refreshing a flagged page guarantees ranking recovery without an A/B test.
  3. *Single 90-day Snapshot:* Seasonal dips or site-wide migrations could register as page-level decay.

### [3:45 – 4:45] Pillar 5: One Concrete Recommendation & Action Playbook
- **Production Implementation:** Rather than a raw probability score, every page is mapped to:
  1. A blended priority score combining model confidence with business impact.
  2. Human-readable reason codes (e.g., `imp_decay_high`, `freshness_risk`).
  3. A concrete editorial action: `refresh`, `refresh_and_review_ctr`, `refresh_and_review_engagement`, `expand_and_refresh`, or `monitor`.
- **Immediate Sprint 1 Target:** 3,576 high-confidence declining pages queued for immediate editorial review.
- **Strict No-Go Checklist:**
  - Never auto-publish rewritten content without human review.
  - Never delete pages based solely on model decline scores.
  - Never guarantee clients that a refresh will restore top-3 rankings.

### [4:45 – 5:00] Wrap-up & Links
- "The full paper is live on GitHub Pages, pipeline receipts are committed, and the notebook runs top-to-bottom."
- **Paper:** https://ErenSnowh.github.io/flyrank-ml-internship/
- **Repo:** https://github.com/ErenSnowh/flyrank-ml-internship

---

## 2. Two Shareable Cuts

### Cut A: Social Post (LinkedIn / X — shareable as-is)

> 🔬 Just published my machine learning research paper from the FlyRank ML Internship:
> *"Can Behavioral Signals Predict Content Decline?"*
>
> **The Problem:** FlyRank manages content portfolios at scale across dozens of clients. But content decays silently in Google search. When an editorial team only has the bandwidth to refresh 50 pages this sprint, which 50 should they choose?
>
> **The Methodology:**
> • 30,000 pseudonymized pages across 32 clients (trailing 90-day search & analytics telemetry).
> • Strict client-holdout validation: evaluated models on clients unseen during training to ensure true generalization.
> • Programmatic zero-leakage assertions: target sources (`trend_direction`, `trend_pct`) and production heuristic outputs were strictly excluded.
>
> **The Key Result:**
> A shallow decision tree achieved **Precision@50 of 0.68** — a **2.8× lift over FlyRank's hand-crafted rule baseline** (0.24) and **1.7× lift over random selection** (0.39).
>
> **What Surprised Me:**
> The hand-crafted rule baseline performed *worse than random guessing* (P@50 = 0.24 vs. 0.39 base rate). Fixed heuristic rules over-indexed on high-impression pages that were actually stable, whereas the learned tree isolated subtle interactions between impression-active days and content maturity.
>
> **The Deliverable:**
> Not a black-box score, but an operational decision-support playbook: 3,576 high-confidence pages mapped to transparent reason codes, concrete editorial action tags (`refresh`, `expand_and_refresh`, `review_ctr`), and a strict "no-go" guardrail list.
>
> 📄 Read the full paper: https://ErenSnowh.github.io/flyrank-ml-internship/
> 💻 Reproducible code & data contract: https://github.com/ErenSnowh/flyrank-ml-internship
>
> Built by Vinayak Pandey (https://allabtme.vercel.app/) during the FlyRank ML Internship.
>
> #MachineLearning #SEO #DataScience #MLOps #SearchAnalytics #AIResearch

---

### Cut B: Employer-Facing Summary (3 sentences — copy-paste ready)

1. **What I built:** I designed and implemented an end-to-end, leakage-free machine learning prioritization pipeline that predicts search visibility decline across a 30,000-page multi-client portfolio using trailing 90-day behavioral and analytics telemetry.
2. **On what data & what it showed:** Evaluated on strict client-holdout splits (holding out 20% of whole clients to prevent identity leakage), my decision-tree model achieved a **Precision@50 of 0.68** — delivering a **2.8× lift over the production hand-crafted rule baseline** (0.24) and correctly surfacing 34 genuinely declining pages in every 50 reviewed.
3. **Why it matters to your team:** The system operationalizes predictions into transparent reason codes, concrete editorial playbooks, and strict deployment guardrails, and is fully documented as a reproducible, [peer-grade research paper](https://ErenSnowh.github.io/flyrank-ml-internship/) with programmatic leakage assertions and zero data leaks.
