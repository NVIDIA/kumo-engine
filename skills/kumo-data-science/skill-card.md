## Description: <br>
Turns a plain-language request for a KumoAI predictive model on relational data in a warehouse, an object store or local files into a confirmed task and graph through a multi-turn intake, generates one notebook or script that trains and predicts with the local Kumo SDK (running it only when asked), and routes graph inspection, explanation, diagnostics, persistence, custom-population and tuning requests to their own Kumo skills. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill with coding agents to build Kumo models for churn, propensity, fraud, recommendation, forecasting and link prediction on relational data in Snowflake, BigQuery, Databricks, S3, GCS or local files: the agent checks the request, proposes the graph and task for confirmation and writes the training-and-prediction notebook, or helps create and debug the jobs that run it. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [Cloud Credentials, Service Account, Other [Snowflake connections.toml entry]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [accept/intake_contract.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-data-science/accept/intake_contract.md) <br>
- [accept/preflight.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-data-science/accept/preflight.md) <br>
- [accept/batch_inference.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-data-science/accept/batch_inference.md) <br>
- [accept/semantic_view.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-data-science/accept/semantic_view.md) <br>
- [validate/atomic_question_check.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-data-science/validate/atomic_question_check.md) <br>
- [hosts/generic/adapter.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-data-science/hosts/generic/adapter.md) <br>
- [hosts/generic/runtime.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-data-science/hosts/generic/runtime.md) <br>
- [NVIDIA/kumo-engine#134 (graph visualization)](https://github.com/NVIDIA/kumo-engine/pull/134) <br>
- [Kumo Engine documentation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview) <br>
- [NVIDIA/kumo-engine](https://github.com/NVIDIA/kumo-engine) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis] <br>
**Output Format:** [Markdown replies (ask-backs, graph proposal, task restatement) and one Jupyter notebook or Python script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates code only; runs it only when the user asks. Writes predictions only to a destination the user confirmed.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative), 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox` environment; evaluator version `1.6.9`; evaluated 2026-11-18). Each task was also attempted without the skill as a baseline. <br>

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
| Overall | 99.1% — baseline ran, but no comparable score was available; uplift unavailable | 94.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 93.8% → 100.0% (+6.2 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 47.5% → 100.0% (+52.5 points) |
| Discoverability | 99.3% — baseline ran, but no comparable score was available; uplift unavailable | 90.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 39.3% → 100.0% (+60.7 points) | 38.8% → 85.9% (+47.1 points) |
| Efficiency | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.1% → 92.5% (-4.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.3% → 90.4% (+7.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 27 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Skill Version(s): <br>
c7554e32 (source: git SHA, committed 2026-11-17); repository pyproject.toml version 0.1.0; SKILL.md build marker SKILL-BUILD: 2026-11-17-53c38923-dirty <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
