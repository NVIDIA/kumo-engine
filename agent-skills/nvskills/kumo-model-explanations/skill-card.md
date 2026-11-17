## Description: <br>
Inspect and interpret gradient attribution, column analysis and sampled explanation subgraphs from a Kumo model trained in a Python notebook. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill with coding agents to read the explanation artifacts of an existing Kumo run, such as which table columns the model relied on and how predictions compare with outcomes across column cohorts, without retraining. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>  

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. See skill body for more details. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [references/artifacts.md](https://github.com/NVIDIA/kumo-engine/blob/main/skills/kumo-model-explanations/references/artifacts.md) <br>
- [NVIDIA/kumo-engine](https://github.com/NVIDIA/kumo-engine) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis] <br>
**Output Format:** [Markdown assessment with inline Python code blocks and plots] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Inspection only; makes no causal claims and starts no fit.] <br>

## Skill Version(s): <br>
Unversioned; published from the main branch of NVIDIA/kumo-engine (source: repository; no skill version, CHANGELOG entry or release tag covers the skill files). The signed copy is identified by the skill.oms.sig file in this directory. <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
