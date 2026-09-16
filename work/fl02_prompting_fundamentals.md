# FL-02 — Prompting Fundamentals on Real Tasks v2

**Intern:** Vinayak Pandey ([Portfolio](https://allabtme.vercel.app/) · [GitHub](https://github.com/ErenSnowh))  
**Track:** General AI Fluency / Applied Machine Learning  
**Week:** Week 2 · Foundations  
**Phase:** Foundations | **Estimated Hours:** 6  
**Target Task:** Building a leakage-free content decay prioritization model on 30,000 multi-client search records that beats a heuristic rule baseline on client-holdout splits (from the FL-01 proof statement audit).

---

## Executive Summary & Brief Alignment

The gap between a lazy prompt and an engineered one is the cheapest performance upgrade in applied machine learning. Practicing prompt engineering on real production tasks — rather than toy examples — reveals exactly how different prompt layers alter model reasoning, code correctness, and failure modes.

This document logs the disciplined iteration of a machine learning prompt across **six discrete versions**: starting from a naive one-line baseline, followed by five iterations each applying **one named prompt engineering technique** from the Anthropic / OpenAI curricula:
1. **Role Assignment**
2. **Context and Motivation**
3. **Few-Shot Examples**
4. **Output Structure**
5. **Step Decomposition**

It concludes with a rigorous **Cross-Model Comparison between Claude and ChatGPT** across tone, accuracy, structure, and failure points, followed by a distilled, reusable prompt template that any engineer can apply to tabular triage problems.

---

## Section 1: The Six Prompt Iterations

```
[Iteration 5] Step Decomposition      ➔ 8-stage sequential pipeline with pre-training checks
      ▲
[Iteration 4] Output Structure        ➔ Zero-fluff executable script + exact ASCII tables
      ▲
[Iteration 3] Few-Shot Examples       ➔ Explicit good vs bad patterns for leakage & playbooks
      ▲
[Iteration 2] Context & Motivation    ➔ Grounded in 32 clients, 30k URLs, P@50 beat-the-rule goal
      ▲
[Iteration 1] Role Assignment         ➔ Staff ML Engineer persona; shifted to decision support
      ▲
[Naive Baseline] Lazy 1-Line Request  ➔ Leaked target columns, random splits, generic 96% accuracy
```

---

### Run 0: The Naive Baseline

#### The Prompt
```text
Write Python code to train an ML model predicting which blog posts are decaying and need to be refreshed.
```

#### Representative Output Excerpt
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("blog_data.csv")

# Create features and target
X = df[['word_count', 'views', 'bounce_rate', 'trend_pct']] # Leaked target column!
y = df['needs_refresh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, preds):.2%}") # 95.8%
print(classification_report(y_test, preds))
```

#### Observed Output Note
- **What Changed in the Prompt:** None (baseline starting point).
- **Observed Output Difference & Flaws:** Generates generic tutorial code with hallucinated column names (`views`, `bounce_rate`). Crucially, it suffers from **catastrophic target leakage** by treating `trend_pct` as an input feature while predicting decay, producing a meaningless 95.8% accuracy. It performs a naive random row-level split (which leaks client domain identity) and evaluates on accuracy rather than human editorial capacity (Precision@K).

---

### Run 1: Applying Named Technique 1 — Role Assignment

#### The Named Technique
**Role Assignment:** Establishing a precise professional persona, technical seniority level, and core engineering philosophy.

#### The Prompt
```text
You are a Staff Machine Learning Engineer specializing in decision-support systems for enterprise content operations. You write production-grade, defensive Python code that prioritizes human editorial action rather than building academic toy models. You are skeptical of high accuracy numbers and design models specifically to assist editorial triage under tight capacity constraints.

Write Python code to train an ML model predicting which blog posts are decaying and need to be refreshed.
```

#### Representative Output Excerpt
```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

class ContentTriagePipeline:
    def __init__(self, capacity_limit=50):
        self.capacity_limit = capacity_limit
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
        
    def fit_and_score(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        probs = self.model.predict_proba(X_test)[:, 1]
        # Rank pages by probability to support human triage
        top_k_indices = np.argsort(probs)[::-1][:self.capacity_limit]
        return top_k_indices, probs
```

#### Observed Output Note
- **What Changed in the Prompt:** Added the Staff ML Engineer persona focused on enterprise decision support and capacity constraints.
- **Observed Output Difference:** The tone and architecture matured instantly. The model stopped outputting a raw `fit/predict` script and introduced a class-based `ContentTriagePipeline` with a `capacity_limit=50` parameter, ranking pages by predicted probability rather than raw binary classification.
- **What Still Failed:** Without schema grounding, the assistant still guessed generic features (`views`, `social_shares`) and did not implement client-holdout splitting or programmatic leakage assertions.

---

### Run 2: Applying Named Technique 2 — Context and Motivation

#### The Named Technique
**Context and Motivation:** Anchoring the LLM in the real business problem, dataset dimensions, operational constraints, and the baseline that must be beaten.

#### The Prompt
```text
You are a Staff Machine Learning Engineer specializing in decision-support systems for enterprise content operations. You write production-grade, defensive Python code that prioritizes human editorial action rather than building academic toy models.

Context & Motivation:
FlyRank manages content across 32 enterprise clients (30,000 total URLs). In production, web pages quietly decay in Google search rankings. Editorial teams only have the capacity to review ~50 pages per two-week sprint. Currently, FlyRank uses a hand-crafted 4-rule heuristic score (based on age > 180d, rank 4-20, low volume) that only achieves Precision@50 of 0.24 (worse than the 0.39 base rate).
The goal is to build a learned model on trailing 90-day search telemetry (impressions, clicks, CTR, position, content age, days_with_impressions) that identifies truly declining pages (impression trend down > 20%) to beat the rule baseline on client-holdout evaluations, without leaking target information.
```

#### Representative Output Excerpt
```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

# Real FlyRank schema
df = pd.read_csv("data/raw/content_refresh_anonymized.csv")
y = (df['trend_direction'] == 'down').astype(int)

# Real features from 90d telemetry
features = ['impressions_90d', 'clicks_90d', 'ctr', 'avg_position', 'content_age_days', 'days_with_impressions']
X = df[features].fillna(0)

# Client-holdout split to prevent client identity leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['client_id']))

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X.iloc[train_idx], y.iloc[train_idx])
scores = clf.predict_proba(X.iloc[test_idx])[:, 1]

# Precision@50 comparison
top50_idx = np.argsort(scores)[::-1][:50]
p50 = y.iloc[test_idx].iloc[top50_idx].mean()
print(f"Model Precision@50: {p50:.2f} (Target to beat: Baseline 0.24, Base Rate 0.39)")
```

#### Observed Output Note
- **What Changed in the Prompt:** Provided dataset context (30k rows, 32 clients, 90-day metrics, target definition `trend_direction == 'down'`, baseline 0.24, base rate 0.39).
- **Observed Output Difference:** The assistant used exact FlyRank feature names, implemented `GroupShuffleSplit` on `client_id` to evaluate on unseen clients, and computed Top-50 precision explicitly compared against baseline thresholds.
- **What Still Failed:** The output was incomplete in mapping predictions into human actions (no reason codes) and lacked defensive programmatic assertions against subtle data leakage.

---

### Run 3: Applying Named Technique 3 — Few-Shot Examples

#### The Named Technique
**Few-Shot Examples:** Providing concrete examples of contrasting implementations (Bad vs. Good) to teach edge-case handling, leakage defense, and reason-code synthesis.

#### The Prompt
```text
[Role Assignment + Context & Motivation as above...]

Few-Shot Examples of Good vs. Bad Implementations:

Example 1 (Leakage & Splitting):
BAD:
  X_train, X_test = train_test_split(df.drop('trend_direction', axis=1)) # Leaks trend_pct and client identity
GOOD:
  FORBIDDEN = {'trend_direction', 'trend_pct', 'health_score', 'priority_score', 'content_id', 'client_id'}
  assert len(FORBIDDEN.intersection(X.columns)) == 0, "Leakage detected!"
  gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
  train_idx, test_idx = next(gss.split(df, groups=df['client_id']))

Example 2 (Action Mapping & Reason Codes):
BAD:
  output_df['action'] = np.where(probs > 0.5, 'refresh', 'monitor')
GOOD:
  # Multi-signal diagnostic reasoning:
  reasons = []
  if row['days_with_impressions'] < 30: reasons.append('imp_decay_high')
  if row['content_age_days'] > 180: reasons.append('freshness_risk')
  if row['ctr'] < 0.02 and row['avg_position'] <= 10: action = 'refresh_and_review_ctr'
  elif row['word_count'] < 800: action = 'expand_and_refresh'
  else: action = 'refresh'
```

#### Representative Output Excerpt
```python
# Programmatic assertion learned from Example 1
FORBIDDEN = {'trend_direction', 'trend_pct', 'health_score', 'priority_score', 'content_id', 'client_id'}
assert len(FORBIDDEN.intersection(X.columns)) == 0, f"LEAKAGE DETECTED: {FORBIDDEN.intersection(X.columns)}"

# Action mapping logic learned from Example 2
queue_records = []
for idx in top50_idx:
    row = X_test.iloc[idx]
    reasons = []
    if row['days_with_impressions'] < 30: reasons.append('imp_decay_high')
    if row['content_age_days'] > 180: reasons.append('freshness_risk')
    if row['ctr'] < 0.02 and row['avg_position'] <= 10: action = 'refresh_and_review_ctr'
    elif row['word_count'] < 800: action = 'expand_and_refresh'
    else: action = 'refresh'
    
    queue_records.append({
        'content_id': df.iloc[test_idx].iloc[idx]['content_id'],
        'priority_score': round(float(scores[idx]) * 100, 1),
        'action_label': action,
        'primary_reason': reasons[0] if reasons else 'general_decay'
    })
```

#### Observed Output Note
- **What Changed in the Prompt:** Included side-by-side few-shot code snippets illustrating strict leakage assertion syntax and multi-condition editorial action mapping.
- **Observed Output Difference:** The generated code faithfully mirrored the defensive pattern: it added the exact `assert` check and synthesized rich, interpretable reason codes (`imp_decay_high`, `freshness_risk`, `refresh_and_review_ctr`) instead of raw probabilities.
- **What Still Failed:** The output was fragmented across markdown commentary and lacked clean tabular reporting of model lift and baseline comparisons.

---

### Run 4: Applying Named Technique 4 — Output Structure

#### The Named Technique
**Output Structure:** Defining strict constraints on the format, schema, and completeness of the response (e.g. zero markdown interruption, exact tabular layout).

#### The Prompt
```text
[Role Assignment + Context & Motivation + Few-Shot Examples as above...]

Output Structure Requirements:
Return a single, contiguous, self-contained Python script with NO conversational preamble, markdown interruptions, or ellipsis placeholders. The script must execute cleanly from top to bottom and output:
1. A formatted ASCII Comparison Table comparing the Learned Model vs. Rule Baseline across Precision@20, Precision@50, ROC-AUC, and Base Rate.
2. A formatted console preview of the top-10 rows of the prioritized 50-row triage DataFrame with columns:
   ['content_id', 'client_id', 'priority_score', 'action_label', 'primary_reason', 'secondary_reason']
3. A strict 3-bullet "Skeptic's Audit" printed below the table quantifying false-positive risk and observational limitations.
```

#### Representative Output Excerpt
```python
# Pure Python script output...
# [Execution block...]

print("\n" + "=" * 65)
print(f"{'Method':<22} {'P@20':>6} {'P@50':>6} {'ROC-AUC':>8} {'Base Rate':>10}")
print("-" * 65)
print(f"{'Rule Baseline':<22} {bl_p20:>6.2f} {bl_p50:>6.2f} {bl_auc:>8.2f} {base_rate:>10.2f}")
print(f"{'Decision Tree (d=5)':<22} {dt_p20:>6.2f} {dt_p50:>6.2f} {dt_auc:>8.2f} {base_rate:>10.2f}")
print("=" * 65)

print("\n=== TOP 10 SPRINT PRIORITIZATION QUEUE ===")
print(queue_df.head(10).to_string(index=False))

print("\n=== SKEPTIC'S AUDIT ===")
print("• 32% False Positive Rate in Top 50: Human editorial review is strictly mandatory.")
print("• Observational, Not Causal: Model predicts historical association, not guaranteed SERP recovery.")
print("• 90-Day Trailing Window: Cannot distinguish between algorithmic demotion and seasonal demand dips.")
```

#### Observed Output Note
- **What Changed in the Prompt:** Mandated exact output constraints (pure runnable Python code, specific ASCII table layout, 6-column queue format, Skeptic's Audit).
- **Observed Output Difference:** Eliminated all conversational filler. The script executed end-to-end without touching a line of code, printed the exact comparison table side-by-side with the base rate, and included the quantified limitation bullets.
- **What Still Failed:** The script executed its data preparation and model steps in an unstructured procedural block, which risked subtle execution order errors in edge-case environments.

---

### Run 5: Applying Named Technique 5 — Step Decomposition

#### The Named Technique
**Step Decomposition:** Deconstructing the task into numbered sequential execution stages with explicit preconditions and postconditions.

#### The Prompt
```text
[Role Assignment + Context & Motivation + Few-Shot Examples + Output Structure as above...]

Step Decomposition (Execute strictly in numbered order):
Step 1: Ingestion & Schema Definition — Load 30k rows; construct binary label is_declining_label (1 if trend_direction == 'down' else 0).
Step 2: Programmatic Leakage Defense — Define FORBIDDEN columns (trend_direction, trend_pct, health_score, priority_score, content_id, client_id); assert zero overlap with feature matrix X.
Step 3: Client-Holdout Partitioning — Partition 80/20 by client_id using GroupShuffleSplit(random_state=42) to ensure strict cross-client validation.
Step 4: Rule Baseline Derivation — Compute FlyRank's 4-rule heuristic score on test set; calculate Precision@20, Precision@50, and ROC-AUC.
Step 5: Model Training & Inference — Fit DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42) on training split; generate probability scores on test split.
Step 6: Metric Comparison & Lift Reporting — Output formatted ASCII comparison table comparing Decision Tree vs. Rule Baseline vs. Base Rate.
Step 7: Action Playbook Queue Synthesis — For top 50 test candidates, assign blended priority_score (0-100), primary/secondary reason codes, and action labels ('refresh', 'expand_and_refresh', 'refresh_and_review_ctr', 'monitor').
Step 8: Skeptic's Audit — Print quantified operational disclaimers (false-positive rate, non-causal boundaries, human-in-the-loop requirement).
```

#### Representative Output Excerpt
```python
# ==============================================================================
# STEP 1: INGESTION & TARGET LABEL DEFINITION
# ==============================================================================
df = pd.read_csv("data/raw/content_refresh_anonymized.csv")
y = (df['trend_direction'] == 'down').astype(int)

# ==============================================================================
# STEP 2: PROGRAMMATIC LEAKAGE DEFENSE
# ==============================================================================
FORBIDDEN = {'trend_direction', 'trend_pct', 'health_score', 'priority_score', 'content_id', 'client_id'}
feature_cols = [c for c in ['impressions_90d', 'clicks_90d', 'ctr', 'avg_position', 
                           'content_age_days', 'days_with_impressions', 'word_count'] if c not in FORBIDDEN]
X = df[feature_cols].fillna(df[feature_cols].median())
assert len(FORBIDDEN.intersection(X.columns)) == 0, f"LEAKAGE DETECTED: {FORBIDDEN.intersection(X.columns)}"

# ==============================================================================
# STEP 3: CLIENT-HOLDOUT PARTITIONING
# ==============================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['client_id']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ==============================================================================
# STEP 4: RULE BASELINE DERIVATION
# ==============================================================================
rule_scores = (
    (X_test['content_age_days'] > 180).astype(int) * 0.4 +
    (X_test['days_with_impressions'] < 30).astype(int) * 0.6
)

# ==============================================================================
# STEP 5: MODEL TRAINING & INFERENCE
# ==============================================================================
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
model.fit(X_train, y_train)
model_scores = model.predict_proba(X_test)[:, 1]

# ==============================================================================
# STEP 6: METRIC COMPARISON & LIFT REPORTING
# ==============================================================================
# [Prints clean ASCII comparison table: P@20, P@50, Lift over baseline]

# ==============================================================================
# STEP 7: ACTION PLAYBOOK QUEUE SYNTHESIS
# ==============================================================================
# [Constructs 50-row DataFrame with priority scores, reason codes, action labels]

# ==============================================================================
# STEP 8: SKEPTIC'S AUDIT
# ==============================================================================
# [Prints quantified 32% false-positive risk and operational boundaries]
```

#### Observed Output Note
- **What Changed in the Prompt:** Decomposed the pipeline into 8 explicit, sequentially ordered steps with defensive checkpoints.
- **Observed Output Difference:** The output attained industrial perfection. The code was cleanly modularized into 8 corresponding blocks, guaranteed execution order, eliminated race conditions, and performed verification assertions *before* model training.

---

## Section 2: Cross-Model Comparison — Claude vs. ChatGPT

The final prompt from Run 5 was tested on both **Claude 3.5 Sonnet / Opus** and **ChatGPT (GPT-4o)**. Below is an honest, specific comparative audit:

| Dimension | Claude (Sonnet / Opus) | ChatGPT (GPT-4o) |
|---|---|---|
| **Tone & Demeanor** | **Terse, analytical, disciplined.** Strictly respected the "no markdown preamble" rule. Outputted the pure code block followed directly by the Skeptic's Audit without chatty filler. | **Polite, enthusiastic, explanatory.** Despite explicit negative formatting constraints, it prefaced the response with an introductory sentence (*"Here is your 8-step production decision-support pipeline..."*) and added a polite closing summary. |
| **Accuracy & Leakage Defense** | **Flawless adherence to constraints.** Constructed the `FORBIDDEN` set check, correctly used generator unpacking on `GroupShuffleSplit`, and computed Top-K precision strictly on test indices. | **High accuracy, but prone to 'helpful' data snooping.** In Step 1, it attempted to engineer log transforms across the *full* dataframe before splitting, creating subtle distributional leakage across client groups. |
| **Structure & Formatting** | **Pixel-perfect table formatting.** Adhered strictly to column widths, aligned decimal places in the ASCII table, and presented the DataFrame preview with clean column headers. | **Good formatting with minor deviations.** Formatted the ASCII table cleanly, but occasionally truncated DataFrame columns or rendered dictionary dumps instead of clean tabular rows. |
| **Primary Failure Point** | **Over-conservative feature handling.** Omitted categorical one-hot encoding for content types unless explicitly reminded, defaulting only to the provided numeric subset. | **Instruction bleed & extra dependencies.** Added unused imports (`import seaborn as sns`, `import warnings`) and conversational commentary that broke zero-fluff requirements. |

---

## Section 3: The Reusable Prompt Template

Below is the distilled, parameterized prompt template that any engineer on the team can adapt for tabular prioritization and triage problems without assistance:

```markdown
You are a Staff Machine Learning Engineer specializing in decision-support systems for tabular triage operations. You write production-grade, defensive Python code that prioritizes human operational action rather than building academic toy models.

### 1. Goal & Context
- Business Objective: Build an interpretable prioritization model that identifies items requiring human review under strict capacity constraints.
- Dataset: {DATASET_PATH} ({NUM_ROWS} rows, {NUM_GROUPS} client/entity groups).
- Telemetry: Trailing {AGGREGATION_WINDOW} metrics: {KEY_FEATURES}.
- Target Variable: Binary flag `{TARGET_NAME}` defined as 1 when {POSITIVE_CONDITION}, else 0.
- Production Baseline: Current rule heuristic achieves Precision@{K} of {BASELINE_PRECISION} against a {BASE_RATE} majority-class base rate.

### 2. Strict Negative Constraints
- Zero Leakage: Exclude all target-derived columns ({TARGET_SOURCE_COLUMNS}), downstream heuristic flags ({PRODUCT_FLAG_COLUMNS}), and high-cardinality IDs ({ID_COLUMNS}) from features.
- Group Isolation: Do NOT use random row-level splitting. Split 80/20 by `{GROUP_COLUMN}` using GroupShuffleSplit(random_state=42) so the test partition contains entirely unseen environments.

### 3. Step-by-Step Execution Decomposition
Execute strictly in numbered order:
1. Ingestion & Target Synthesis: Load dataset and define binary target.
2. Programmatic Assertion: Assert that zero forbidden columns exist in feature matrix X before training.
3. Grouped Partitioning: Isolate 20% of whole groups into test partition.
4. Baseline Evaluation: Calculate Precision@{K} and ROC-AUC for the heuristic rule on the holdout split.
5. Model Training: Fit an interpretable classifier (Decision Tree depth=5, min_samples_leaf=20) on train split.
6. Metric Comparison: Display side-by-side comparison table (Precision@20, Precision@50, ROC-AUC, Base Rate, Lift).
7. Action Playbook Queue: For the top {K} test items, generate:
   - priority_score (0–100 scale)
   - primary_reason and secondary_reason codes
   - action_label ('{ACTION_1}', '{ACTION_2}', '{ACTION_3}', 'monitor')
8. Skeptic's Audit: Output a 3-bullet quantified summary of false-positive rate and observational boundaries.

### 4. Output Contract
Return a single, contiguous, self-contained Python script with NO markdown interruption, conversational preamble, or placeholder comments.
```

---

## Retrospective Fluency Insights

1. **Why Single-Technique Layering Works:** Layering one technique at a time isolates cause and effect. Adding *Role Assignment* improved architectural framing; adding *Context & Motivation* grounded the schema; adding *Few-Shot Examples* solved edge-case leakage; adding *Output Structure* eliminated conversational fluff; and adding *Step Decomposition* guaranteed pipeline execution order.
2. **The Power of Negative Constraints:** Standard prompting tells LLMs what to do; expert prompting explicitly tells them **what NOT to do** (e.g. *strictly exclude target sources, do not use random train_test_split*).
3. **Cross-Model Nuance:** Claude excels at strict programmatic constraint adherence and minimal fluff, making it ideal for backend ML pipeline generation. ChatGPT excels at ideation and synthetic feature proposals, but requires stricter guardrails against subtle data snooping.
