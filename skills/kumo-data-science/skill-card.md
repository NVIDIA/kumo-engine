## Description: <br>
Use for KumoAI graphs and predictive models on relational data in a warehouse, an object store or local files: inspect, visualize or export a graph's schema and relationships; train a model, score data with a trained one, explain its predictions, diagnose its training, publish, reload or retain it, train or score on custom labels or rows, or tune it. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill to turn a plain-language predictive request (churn, propensity, fraud, recommendation, forecasting, link prediction) on relational data into a confirmed KumoAI task and graph, then generate one runnable notebook or script that trains and predicts, or scores with a saved model; it runs the artifact only on explicit request. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Intake contract](accept/intake_contract.md) <br>
- [Preflight & graph gate](accept/preflight.md) <br>
- [Batch inference](accept/batch_inference.md) <br>
- [Semantic view resolution](accept/semantic_view.md) <br>
- [Atomic question check](validate/atomic_question_check.md) <br>
- [Generic host adapter](hosts/generic/adapter.md) <br>
- [Generic host runtime](hosts/generic/runtime.md) <br>
- [NVIDIA/kumo-engine#134 (graph visualization)](https://github.com/NVIDIA/kumo-engine/pull/134) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Files] <br>
**Output Format:** [Markdown proposals and confirmations, plus one Python notebook or script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates one artifact with exactly one blocking fit call; executes it only on explicit user request; predictions are written only to a confirmed destination.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative); 1 attempt per task; each task attempt ran in its own isolated sandbox pod (environment `k8s-sandbox`). Evaluator version `1.6.10`, evaluation date 2026-11-20. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.2% — baseline ran, but no comparable score was available; uplift unavailable | 94.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 81.3% → 100.0% (+18.7 points) |
| Correctness | 52.5% → 100.0% (+47.5 points) | 42.5% → 95.0% (+52.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 35.6% → 100.0% (+64.4 points) | 35.3% → 91.3% (+56.0 points) |
| Efficiency | 96.2% — baseline ran, but no comparable score was available; uplift unavailable | 95.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.3% → 92.4% (-4.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.9% → 90.5% (+7.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml); skill build marker SKILL-BUILD: 2026-11-19-bef5ab4e-dirty (source: SKILL.md) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
