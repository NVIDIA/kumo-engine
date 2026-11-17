## Description: <br>
Turns a plain-language request for a KumoAI predictive model on relational data into a confirmed task and graph through a multi-turn intake, generates one notebook or script that trains and predicts with the local Kumo SDK, and routes inspection, persistence, population and tuning requests to their own Kumo skills. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill with coding agents to build Kumo models for churn, propensity, fraud, recommendation, forecasting and link prediction on data in Snowflake, BigQuery, Databricks, S3, GCS or local files: the agent checks the request, proposes the graph for confirmation and writes the training-and-prediction notebook. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [Cloud Credentials, OAuth Token, Other [Snowflake connections.toml entry]] <br>  

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. See skill body for more details. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [accept/intake_contract.md](accept/intake_contract.md) <br>
- [accept/preflight.md](accept/preflight.md) <br>
- [accept/batch_inference.md](accept/batch_inference.md) <br>
- [validate/atomic_question_check.md](validate/atomic_question_check.md) <br>
- [hosts/generic/adapter.md](hosts/generic/adapter.md) <br>
- [hosts/generic/runtime.md](hosts/generic/runtime.md) <br>
- [NVIDIA/kumo-engine](https://github.com/NVIDIA/kumo-engine) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis] <br>
**Output Format:** [Markdown replies (ask-backs, graph proposal, task restatement) and one Jupyter notebook or Python script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates code only; runs it only when the user asks. Writes predictions only to a destination the user confirmed.] <br>

## Skill Version(s): <br>
Unversioned; published from the main branch of NVIDIA/kumo-engine (source: repository; no skill version, CHANGELOG entry or release tag covers the skill files). The signed copy is identified by the skill.oms.sig file in this directory. <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
