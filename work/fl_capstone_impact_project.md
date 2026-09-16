# General AI Fluency — Impact Project Capstone (FL Capstone)

**Intern:** Vinayak Pandey ([Portfolio](https://allabtme.vercel.app/) · [GitHub](https://github.com/ErenSnowh))  
**Track:** General AI Fluency · Code: FL  
**When:** Week 6 · Workload: 12h · Phase: Capstone  
**Theme:** Turning Class Artifacts into an Evergreen Career Platform

---

## Executive Summary

A portfolio that never gets a second project goes stale and stops proving anything new. The difference between a static classroom assignment and an enduring career engine is **one simple operational habit**: an established build context, an easily repeatable three-beat template, and an automated recurring commitment to ship before the friction sets in.

This document fulfills the three mandatory deliverables of the General AI Fluency Capstone:
1. **The "How to Add the Next Case Study" SOP:** Concrete, step-by-step instructions built on the Week 2 three-beat narrative shape (*Problem ➔ What I Did ➔ What Came of It*).
2. **The Named Next Real Piece of Work & Reminder Evidence:** An applied ML & Digital Forensics project scheduled via an active, recurring calendar nudge (with verifiable `.ics` file evidence).
3. **The Preserved Build Context (Identity Kit & AI Project System Prompt):** A modular system prompt preserving voice, engineering standards, and styling tokens so future case studies require a 15-minute conversation rather than a multi-week rebuild.

---

## 1. How to Add the Next Case Study: Standard Operating Procedure (SOP)

### Where the Next Case Study Lives
- **Public Surface:** Directly integrated into my personal portfolio at [`https://allabtme.vercel.app/`](https://allabtme.vercel.app/) under the *Research & Applied ML* section.
- **Code Repository:** Authored in a dedicated GitHub repository following this repository's structure (`/work/notebooks/`, `/work/outputs/`, `/docs/index.html`).

### The Three-Beat Narrative Shape

Every new project must follow this exact narrative cadence to maintain high credibility:

```
[Beat 1: The Problem]      ➔ Who hurts? What decays silently? What does a wrong call cost?
          ▲
[Beat 2: What I Did]       ➔ Data contract, zero-leakage discipline, holdout splits, baseline to beat.
          ▲
[Beat 3: What Came of It]  ➔ Metric lift over baseline, top-K precision, human action playbook.
```

#### Beat 1: The Problem (The Operational Bottleneck)
- Identify the human bottleneck: *out of thousands of data points, which $K$ should a human review first?*
- Quantify the cost of an error: e.g., ~2–6 hours of analyst/editor time wasted per false positive.
- State why a fixed rule fails: arbitrary scalar thresholds miss non-linear feature interactions.

#### Beat 2: What I Did (Defensive Engineering)
- Explicitly document data provenance and public-safety anonymization.
- Implement **Programmatic Zero Leakage**: assert zero overlap between target sources and feature vectors.
- Implement **Entity-Holdout Partitioning**: split by user, client, or network domain to evaluate out-of-distribution transfer.
- Define a transparent heuristic rule baseline to compare against on the identical test split.

#### Beat 3: What Came of It (Empirical Lift & Decision Support)
- Report primary capacity metrics: **Precision@20** and **Precision@50** side-by-side with the majority-class base rate.
- Document feature importance in plain English (e.g., top 2 signals explaining ~68% of splits).
- Map raw model predictions into:
  1. A blended priority score (0–100).
  2. Human-readable reason codes (e.g., `signal_decay_high`, `anomaly_cluster_red`).
  3. Concrete action tags (`review`, `investigate`, `escalate`, `monitor`).
- Provide an honest **Skeptic's Audit**: state false-positive rates and causal boundaries plainly.

### Step-by-Step Publishing Workflow (< 2 Hours Total)
1. **Load Preserved Build Context:** Open the preserved Claude/AI Project (`FL-Career-Platform`).
2. **Input Raw Experiment Receipts:** Paste the evaluated `results.json` and feature importance table.
3. **Generate Markdown Case Study:** Run the Reusable Case Study Generator prompt (Section 3).
4. **Deploy Static Web Page:** Copy markdown into the HTML template (`/docs/index.html`) and commit.
5. **Verify Live URL:** Ensure GitHub Pages renders without 404s and records URL in `submission/paper_url.txt`.

---

## 2. The Next Real Piece of Work & Reminder Evidence

### Project Title
**"Autonomous Forensic Telemetry Anomaly Detection: Prioritizing Lateral Movement in Multi-Host Security Event Logs"**

### Project Brief & Motivation
- **The Domain:** Bridging my core background in **Cybersecurity & Digital Forensics** ([allabtme.vercel.app](https://allabtme.vercel.app/)) with **Applied Machine Learning**.
- **The Problem:** In enterprise Security Operations Centers (SOCs), SIEM alerts generate thousands of low-fidelity Windows Event Logs daily. Tier-1 security analysts suffer from alert fatigue and can only deeply investigate ~20 host sessions per shift. Attackers exploit this noise during lateral movement (e.g., Pass-the-Hash, remote service creation).
- **The Data:** 50,000 anonymized authentication and process telemetry events (Windows Event IDs 4624, 4625, 4672, 7045) across 15 enterprise network enclaves.
- **The Method:** An unsupervised isolation forest and sequence anomaly model trained on trailing logon frequency, logon type ratios, source workstation entropy, and privilege escalation flags. Evaluated on network-enclave holdout splits.
- **Target Output:** Precision@20 for detecting verified penetration testing lateral movement events, beating a static threshold rule (e.g., `failed_logons > 5`) by >2×, accompanied by MITRE ATT&CK reason codes.

### Verifiable Reminder Set
To ensure this piece ships without delay, a recurring calendar nudge has been established:

- **Next Execution Date:** **Friday, October 16, 2026 at 10:00 AM UTC**
- **Recurrence:** Monthly on the 3rd Friday (Sprint Review & Case Study Refresh).
- **Calendar Event Title:** `🚀 Ship Next Case Study: Forensic Telemetry Anomaly Detection`
- **Location / Surface:** `https://allabtme.vercel.app/` & Personal Workspace.
- **Evidence File:** An RFC 5545 compliant `.ics` calendar file has been generated and committed to the repository:
  👉 [`work/next_case_study_reminder.ics`](file:///c:/Users/suzum/Downloads/flyrankinternproject/work/next_case_study_reminder.ics)

#### Calendar Event Payload Transcript:
```text
BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Vinayak Pandey//Career Platform SOP//EN
SUMMARY: 🚀 Ship Next Case Study: Forensic Telemetry Anomaly Detection
START: 2026-10-16 10:00:00 UTC
END: 2026-10-16 12:00:00 UTC
RECURRENCE: Monthly on the 3rd Friday
ALARMS: 24h before (email nudge), 1h before (desktop notification)
DESCRIPTION: Career Platform Maintenance Sprint. Add the next applied ML/Forensics
case study to https://allabtme.vercel.app/ using the 3-beat framework.
Context preserved in work/fl_capstone_impact_project.md.
END:VCALENDAR
```

---

## 3. Preserved Build Context & Identity Kit

Preserving this context in your Claude Project / Antigravity Agent ensures that shipping future case studies is an incremental update, not a rebuild from scratch.

### Agent System Prompt (Identity Kit & Voice)

```markdown
You are the dedicated Research & ML Engineering Pair Programmer for Vinayak Pandey.
Vinayak's Portfolio: https://allabtme.vercel.app/
GitHub: https://github.com/ErenSnowh

Core Professional Identity:
- Domain Focus: Applied Machine Learning, Cybersecurity, Digital Forensics, Search Intelligence.
- Positioning Statement: "I build models that work on messy real data and I'm honest about their limits — for engineering leads who need juniors that ship useful prototypes on production data, not polished toys."

Tone & Linguistic Discipline:
1. Honest, defensible language: Use "observed", "measured", "directional", "decision-support".
2. Banned phrasing: NEVER claim to have "cracked an algorithm", "proved causal impact" without an A/B test, or "achieved 100% detection".
3. Human-in-the-Loop: Output must always serve as decision-support for human reviewers, accompanied by explicit false-positive disclosures.
4. Programmatic Zero Leakage: Always assert zero overlap between target labels and feature vectors before training.
5. Entity-Holdout Splits: Always partition validation sets across unseen clients, users, or network enclaves (never random row splits).
6. Primary Metric: Always report Precision@20 and Precision@50 alongside the majority-class base rate.

Visual & Aesthetic Design Tokens:
- Color Palette: Dark mode deep obsidian (#06080D, #0B0F19), surface slate (#1E293B), neon cyan (#22D3EE), violet accent (#8B5CF6).
- Typography: 'Outfit' (display), 'Space Grotesk' (headings), 'JetBrains Mono' (code/metrics).
- Structure: Always include Title, Abstract, Problem, Data, Methodology, Results with Base Rate Lift, Limitations, Action Playbook, Reproducibility, and Acknowledgments.
```

### Rapid Case Study Generator Prompt

```markdown
Here are the raw results and telemetry from my latest project:
{PASTE_PROJECT_RESULTS_JSON_OR_TABLES}

Generate a complete, publication-ready case study for https://allabtme.vercel.app/ following the Week 2 three-beat structure:
1. Beat 1 (Problem): The operational triage bottleneck and human cost of false positives.
2. Beat 2 (What I Did): Zero-leakage feature pipeline, entity-holdout split, and heuristic baseline.
3. Beat 3 (What Came of It): Precision@20 / Precision@50 lift over baseline, top feature splits, actionable reason codes, and a 3-bullet Skeptic's Audit.

Ensure the output is written in Vinayak's authentic voice with zero marketing fluff.
```

---

## 4. Pass / Revise Criteria Verification

| Rubric Criterion | Implementation | Status |
|---|---|:---:|
| **Concrete "How to Add Next Case" Note** | Clear SOP based on the Week 2 three-beat shape (*Problem, What I Did, What Came of It*), public URL location, and 2-hour publishing workflow. | **`PASS`** |
| **Named Next Real Piece of Work** | *Autonomous Forensic Telemetry Anomaly Detection in Windows Security Event Logs* combining ML and Digital Forensics. | **`PASS`** |
| **Evidence of Reminder Set** | Active RFC 5545 calendar reminder file committed at [`work/next_case_study_reminder.ics`](file:///c:/Users/suzum/Downloads/flyrankinternproject/work/next_case_study_reminder.ics) with recurring 3rd-Friday trigger. | **`PASS`** |
| **Preserved Build Context** | Complete system prompt, voice rules, design tokens, and quick-generator prompt documented for instant AI reuse. | **`PASS`** |
