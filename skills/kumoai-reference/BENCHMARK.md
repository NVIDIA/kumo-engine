# Skill Benchmark: kumoai-reference

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `kumoai-reference`
- Validation status: `passed`
- Evaluation date: 2026-11-23
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:70c6edab12642418c73b37fd93ff86ac13f4ce8f782eac415e5aa695a3c59505` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| kumoai-reference | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `kumoai-reference`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.1% — baseline ran, but no comparable score was available; uplift unavailable | 91.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 80.0% → 100.0% (+20.0 points) |
| Correctness | 72.0% → 100.0% (+28.0 points) | 96.0% → 100.0% (+4.0 points) |
| Discoverability | 98.8% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 62.2% → 80.0% (+17.8 points) | 68.9% → 81.6% (+12.7 points) |
| Efficiency | 91.5% — baseline ran, but no comparable score was available; uplift unavailable | 86.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.7% → 88.0% (-8.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 79.2% → 87.6% (+8.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,355,484 | 749,745 | +605,739 | +80.79% | skill 5/5; base 5/5 |
| claude-code | kref-01 | 121,585 | 35,643 | +85,942 | +241.12% | skill 1/1; base 1/1 |
| claude-code | kref-02 | 460,738 | 186,914 | +273,824 | +146.50% | skill 1/1; base 1/1 |
| claude-code | kref-03 | 232,928 | 270,869 | -37,941 | -14.01% | skill 1/1; base 1/1 |
| claude-code | kref-04 | 392,790 | 109,586 | +283,204 | +258.43% | skill 1/1; base 1/1 |
| claude-code | kref-05 | 147,443 | 146,733 | +710 | +0.48% | skill 1/1; base 1/1 |
| codex | All cases | 640,454 | 953,631 | -313,177 | -32.84% | skill 5/5; base 5/5 |
| codex | kref-01 | 68,761 | 29,521 | +39,240 | +132.92% | skill 1/1; base 1/1 |
| codex | kref-02 | 237,048 | 522,373 | -285,325 | -54.62% | skill 1/1; base 1/1 |
| codex | kref-03 | 118,328 | 82,811 | +35,517 | +42.89% | skill 1/1; base 1/1 |
| codex | kref-04 | 118,577 | 51,584 | +66,993 | +129.87% | skill 1/1; base 1/1 |
| codex | kref-05 | 97,740 | 267,342 | -169,602 | -63.44% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,995,938 | 1,703,376 | +292,562 | +17.18% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 15 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/graph_construction.md and references/kumo_finetuning_sdk.md:
  "## Concept" in references/graph_construction.md (lines 28-51)
  vs "## Step 4: build the graph" in references/graph_construction.md (lines 188-208)
  vs "# for tables that exist behind it." in references/graph_construction.md (lines 364-397)
  vs "## 5. Graph" in references/kumo_finetuning_sdk.md (lines 355-392)
  vs "# 3. Graph — list of tables, Edge objects, explicit validate." in references/kumo_finetuning_sdk.md (lines 716-725) (`references/graph_construction.md:28`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/graph_construction.md and references/kumo_finetuning_sdk.md:
  "### Defining tables" in references/graph_construction.md (lines 106-143)
  vs "## 4. Tables" in references/kumo_finetuning_sdk.md (lines 308-354) (`references/graph_construction.md:106`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/kumo_finetuning_sdk.md and references/pql_writing.md and references/task_journeys.md:
  "## 6. Predictive query" in references/kumo_finetuning_sdk.md (lines 393-414)
  vs "## Task type is inferred from query shape" in references/pql_writing.md (lines 202-220)
  vs "## SDK usage" in references/pql_writing.md (lines 438-471)
  vs "### Required pre-flight, before any training step" in references/pql_writing.md (lines 472-490)
  vs "## Classification and regression" in references/task_journeys.md (lines 9-26) (`references/kumo_finetuning_sdk.md:393`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/graph_construction.md and references/troubleshooting_signals.md:
  "## Step 5: time columns and temporal correctness" in references/graph_construction.md (lines 238-249)
  vs "### Symptom: an event table has no time column" in references/troubleshooting_signals.md (lines 146-156) (`references/graph_construction.md:238`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/kumoai-reference/SKILL.md`)
- 14 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
