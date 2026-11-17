## Description: <br>
Publish and reload Kumo models through the Snowflake registry, reconstruct compatible scoring inputs, and retain or retrieve run diagnostics and output files through a Snowflake stage. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill with coding agents to publish a Kumo model during its fit, reload an explicit registry version or retained bundle for scoring, and keep selected run files on a Snowflake stage after the notebook session ends. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [Snowflake connection or Snowpark session of the user's notebook]] <br>  

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. See skill body for more details. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [references/reuse.md](references/reuse.md) <br>
- [references/retention.md](references/retention.md) <br>
- [scripts/retain_artifacts.py](scripts/retain_artifacts.py) <br>
- [NVIDIA/kumo-engine](https://github.com/NVIDIA/kumo-engine) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis] <br>
**Output Format:** [Markdown with inline Python code blocks; an archive with a manifest of byte lengths and SHA256 digests when artifacts are retained] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Loading a model does not authorize training, replacing a registry version or overwriting a destination.] <br>

## Skill Version(s): <br>
Unversioned; published from the main branch of NVIDIA/kumo-engine (source: repository; no skill version, CHANGELOG entry or release tag covers the skill files). The signed copy is identified by the skill.oms.sig file in this directory. <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
