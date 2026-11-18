## Description: <br>
Routes KumoAI graph and predictive-modeling requests on relational data in a warehouse, an object store or local files, and for a new model confirms the data, outcome, time window, inferred tables and task through a gated intake before generating one train-then-predict notebook or script, which it runs only when asked. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers, data scientists and ML engineers use this skill to turn plain-language predictive requests (churn, propensity, fraud, recommendation, forecasting, link prediction) on relational data into a confirmed KumoAI task and graph and one runnable notebook or script, and to route graph inspection, explanation, diagnostics, persistence, custom-population, tuning and job requests to their dedicated skills. <br>

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
- [Preflight and graph gate](accept/preflight.md) <br>
- [Batch inference](accept/batch_inference.md) <br>
- [Semantic views](accept/semantic_view.md) <br>
- [Atomic question check](validate/atomic_question_check.md) <br>
- [Generic host adapter](hosts/generic/adapter.md) <br>
- [Generic host runtime](hosts/generic/runtime.md) <br>
- [Kumo Engine documentation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview) <br>
- [kumo-engine on PyPI](https://pypi.org/project/kumo-engine/) <br>
- [NVIDIA/kumo-engine#134 (graph visualization)](https://github.com/NVIDIA/kumo-engine/pull/134) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Files] <br>
**Output Format:** [Markdown replies (graph and task proposals, ask-backs) and one generated Jupyter notebook or Python script using the kumo.engine API] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates code only until the user confirms the graph and task; runs the artifact only on explicit request] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative), 1 attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`), compared against a no-skill baseline. Evaluator version `1.6.10`; evaluated 2026-11-18. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 98.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 52.5% → 100.0% (+47.5 points) | 47.5% → 92.5% (+45.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.6% → 100.0% (+57.4 points) | 38.8% → 80.4% (+41.6 points) |
| Efficiency | 93.4% — baseline ran, but no comparable score was available; uplift unavailable | 94.0% — baseline ran, but no comparable score was available; uplift unavailable |

Baseline is the same task attempted without the target skill. Scores are rounded to one decimal; uplift is derived from the displayed scores and shown in percentage points. A partial dimension was calculated from only the available configured signals.

#### Comparison details

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.0% → 92.4% (-4.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.2% → 90.3% (+7.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

**Overall verdict: PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml); skill build marker 2026-11-17-5026ce47-dirty (source: SKILL.md); git 7cbda7af (committed 2026-11-18) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
