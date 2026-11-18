# Skill Benchmark: kumoai-reference

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `kumoai-reference`
- Validation status: `passed`
- Evaluation date: 2026-11-18
- Evaluator version: `1.6.9`
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
| kumoai-reference | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| kumoai-reference | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

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
| Overall | 93.4% — baseline ran, but no comparable score was available; uplift unavailable | 94.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 80.0% → 100.0% (+20.0 points) | 90.0% → 100.0% (+10.0 points) |
| Correctness | 64.0% → 100.0% (+36.0 points) | 96.0% → 100.0% (+4.0 points) |
| Discoverability | 98.8% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.2% → 80.7% (+30.5 points) | 68.9% → 91.6% (+22.7 points) |
| Efficiency | 87.7% — baseline ran, but no comparable score was available; uplift unavailable | 88.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 92.0% → 88.6% (-3.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 80.8% → 86.5% (+5.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,201,501 | 1,365,680 | -164,179 | -12.02% | skill 5/5; base 5/5 |
| claude-code | kref-01 | 121,639 | 35,557 | +86,082 | +242.10% | skill 1/1; base 1/1 |
| claude-code | kref-02 | 380,338 | 187,276 | +193,062 | +103.09% | skill 1/1; base 1/1 |
| claude-code | kref-03 | 233,155 | 549,777 | -316,622 | -57.59% | skill 1/1; base 1/1 |
| claude-code | kref-04 | 318,106 | 109,559 | +208,547 | +190.35% | skill 1/1; base 1/1 |
| claude-code | kref-05 | 148,263 | 483,511 | -335,248 | -69.34% | skill 1/1; base 1/1 |
| codex | All cases | 915,158 | 605,989 | +309,169 | +51.02% | skill 5/5; base 5/5 |
| codex | kref-01 | 64,938 | 46,310 | +18,628 | +40.22% | skill 1/1; base 1/1 |
| codex | kref-02 | 188,313 | 249,632 | -61,319 | -24.56% | skill 1/1; base 1/1 |
| codex | kref-03 | 135,073 | 75,281 | +59,792 | +79.43% | skill 1/1; base 1/1 |
| codex | kref-04 | 140,779 | 51,536 | +89,243 | +173.17% | skill 1/1; base 1/1 |
| codex | kref-05 | 386,055 | 183,230 | +202,825 | +110.69% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,116,659 | 1,971,669 | +144,990 | +7.35% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/graph_construction.md and references/kumo_finetuning_sdk.md:
  "### Defining tables" in references/graph_construction.md (lines 106-143)
  vs "## 4. Tables" in references/kumo_finetuning_sdk.md (lines 308-354) (`references/graph_construction.md:106`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/kumo_finetuning_sdk.md and references/pql_writing.md and references/task_journeys.md:
  "## 6. Predictive query" in references/kumo_finetuning_sdk.md (lines 393-414)
  vs "## Task type is inferred from query shape" in references/pql_writing.md (lines 202-220)
  vs "## SDK usage" in references/pql_writing.md (lines 438-471)
  vs "### Required pre-flight, before any training step" in references/pql_writing.md (lines 472-490)
  vs "## Classification and regression" in references/task_journeys.md (lines 9-26) (`references/kumo_finetuning_sdk.md:393`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/kumoai-reference/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/kumoai-reference/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in outputs.md (`skills/kumoai-reference/SKILL.md`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
