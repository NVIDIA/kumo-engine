# Skill Benchmark: kumo-data-science

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `kumo-data-science`
- Validation status: `passed`
- Evaluation date: 2026-11-18
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: `sha256:f72dc6983fe9c6ff4f46b5d40bebd4ceb50356c04e83ebc88dc2cd383486d627` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| kumo-data-science | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumo-data-science | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumo-data-science | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumo-data-science | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| kumo-data-science | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `kumo-data-science`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 98.4% — baseline ran, but no comparable score was available; uplift unavailable | 95.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 57.5% → 100.0% (+42.5 points) | 50.0% → 97.5% (+47.5 points) |
| Discoverability | 99.3% — baseline ran, but no comparable score was available; uplift unavailable | 92.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.4% → 97.9% (+56.5 points) | 36.2% → 93.8% (+57.6 points) |
| Efficiency | 94.5% — baseline ran, but no comparable score was available; uplift unavailable | 93.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.0% → 92.6% (-4.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 80.4% → 90.3% (+9.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,456,211 | 933,207 | +523,004 | +56.04% | skill 8/8; base 8/8 |
| claude-code | kds-01 | 225,387 | 107,911 | +117,476 | +108.86% | skill 1/1; base 1/1 |
| claude-code | kds-02 | 183,714 | 201,375 | -17,661 | -8.77% | skill 1/1; base 1/1 |
| claude-code | kds-03 | 229,800 | 158,004 | +71,796 | +45.44% | skill 1/1; base 1/1 |
| claude-code | kds-04 | 222,519 | 108,540 | +113,979 | +105.01% | skill 1/1; base 1/1 |
| claude-code | kds-05 | 217,354 | 107,133 | +110,221 | +102.88% | skill 1/1; base 1/1 |
| claude-code | kds-06 | 169,879 | 107,398 | +62,481 | +58.18% | skill 1/1; base 1/1 |
| claude-code | kds-07 | 171,634 | 107,252 | +64,382 | +60.03% | skill 1/1; base 1/1 |
| claude-code | kds-08 | 35,924 | 35,594 | +330 | +0.93% | skill 1/1; base 1/1 |
| codex | All cases | 601,311 | 2,529,905 | -1,928,594 | -76.23% | skill 8/8; base 8/8 |
| codex | kds-01 | 70,062 | 45,543 | +24,519 | +53.84% | skill 1/1; base 1/1 |
| codex | kds-02 | 96,604 | 1,481,282 | -1,384,678 | -93.48% | skill 1/1; base 1/1 |
| codex | kds-03 | 122,113 | 819,648 | -697,535 | -85.10% | skill 1/1; base 1/1 |
| codex | kds-04 | 94,265 | 41,384 | +52,881 | +127.78% | skill 1/1; base 1/1 |
| codex | kds-05 | 71,914 | 30,254 | +41,660 | +137.70% | skill 1/1; base 1/1 |
| codex | kds-06 | 83,337 | 41,036 | +42,301 | +103.08% | skill 1/1; base 1/1 |
| codex | kds-07 | 49,343 | 57,280 | -7,937 | -13.86% | skill 1/1; base 1/1 |
| codex | kds-08 | 13,673 | 13,478 | +195 | +1.45% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,057,522 | 3,463,112 | -1,405,590 | -40.59% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and accept/preflight.md:
  "### Node 1c — Preflight & graph gate" in SKILL.md (lines 178-194)
  vs "## 5. Column availability" in accept/preflight.md (lines 157-171) (`SKILL.md:178`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across validate/atomic_question_check.md and validate/examples/bad_request.md:
  "### Fail 1 — Multiple targets bundled" in validate/atomic_question_check.md (lines 42-50)
  vs "## 6. Bundled goals → REJECT" in validate/examples/bad_request.md (lines 64-68) (`validate/atomic_question_check.md:42`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within accept/batch_inference.md:
  "## Node 1a — fields" in accept/batch_inference.md (lines 7-29)
  vs "### `batch_inference` — required fields (ask if missing)" in accept/batch_inference.md (lines 30-42)
  vs "### `batch_inference` — resolved, defaulted, or inferred (never ask)" in accept/batch_inference.md (lines 43-54) (`accept/batch_inference.md:7`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across accept/intake_contract.md and validate/examples/good_request.md:
  "## Canonical good example (`train`)" in accept/intake_contract.md (lines 137-148)
  vs "## Input" in validate/examples/good_request.md (lines 3-12)
  vs "## Node 1a — Required fields: PASS" in validate/examples/good_request.md (lines 13-19) (`accept/intake_contract.md:137`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/kumo-data-science/SKILL.md`)
- 25 additional finding(s) are available in the full evaluation artifacts.

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
