## Description: <br>
Reference for translating a validated ML request into KumoAI graph configuration, PQL (predictive query language) and Kumo SDK code. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill, normally through the kumo-data-science skill, so a coding agent writes Kumo graph definitions, predictive queries and training and prediction code that match the shipped Kumo SDK. <br>

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
- [references/graph_construction.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumoai-reference/references/graph_construction.md) <br>
- [references/pql_writing.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumoai-reference/references/pql_writing.md) <br>
- [references/kumo_finetuning_sdk.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumoai-reference/references/kumo_finetuning_sdk.md) <br>
- [references/task_journeys.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumoai-reference/references/task_journeys.md) <br>
- [references/outputs.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumoai-reference/references/outputs.md) <br>
- [references/troubleshooting_signals.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumoai-reference/references/troubleshooting_signals.md) <br>
- [NVIDIA/kumo-engine](https://github.com/NVIDIA/kumo-engine) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis] <br>
**Output Format:** [Markdown with inline Python and PQL code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates code only; does not run jobs or interpret metrics.] <br>

## Skill Version(s): <br>
Unversioned; published from the main branch of NVIDIA/kumo-engine (source: repository; no skill version, CHANGELOG entry or release tag covers the skill files). The signed copy is identified by the skill.oms.sig file in this directory. <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
