# Maintaining repository documentation

## Canonical sources

| Topic | Authored source |
| --- | --- |
| Product entry | `README.md` |
| System boundaries | `ARCHITECTURE.md` |
| Setup, testing, review, conventions, generation | `docs/development/` and component guides |
| Website user guides/reference | [kumo-ai/docs](https://github.com/kumo-ai/docs), maintained and published separately |
| Application deployment requirements | `helm/kumo-helm/docs/` |
| Helm support boundary | `helm/README.md` |
| Agent routing and local constraints | Root and selected nested `AGENTS.md` |
| Agent procedures | Existing root and component `.agents/skills/` directories |

Use domain ownership from [CODEOWNERS](../../CODEOWNERS). Update a guide in the
same change that alters its documented behavior. Verify commands against build
scripts and contracts against consumers; a timestamp or passing link check
is not evidence of semantic accuracy.

## Validation

The checker discovers all tracked and untracked, unignored Markdown files
through Git; new guides do not need registration. Relative links in symlinked
docs resolve from the authored file. Broken links in the imported
`packages/diskgraph` documentation remain warnings; other broken local links
fail the check. Fix DiskGraph documentation directly in this repository.

[agent-contract.json](agent-contract.json) records instruction scopes and
compatibility adapters. Imported packages retain their own instruction layouts.

`skill_commands` lists the root skills whose Claude command adapters are
maintained by this contract. Each listed adapter must exist and reference its
canonical skill. Other project commands are allowed, and adding a skill does
not require a command adapter. Ordinary command content remains subject to
public-content review.

```bash
python3 -m unittest discover -s test/scripts/docs -p 'test_*.py'
python3 scripts/docs/check_repository_docs.py --root . --check
bash .agents/skills/code-ui/scripts/generate-catalog-docs.sh --check
```

The checker validates local links, heading anchors, instruction imports, and
skill adapters. It skips fenced examples and does not fetch external URLs.
A bundle-layout fixture checks authored Kumo links after assembly without
rendering or installing workloads. Website validation runs in `kumo-ai/docs`.

These checks do not replace secret/internal-reference checks, asset review,
or the source-history audit before publication.

Do not put secrets, private URLs, customer logs, or restricted evidence into
public migration reports. Keep authored detail separate from generated indexes
so regeneration cannot overwrite human-maintained explanations.

## Publication checks

| Command | Purpose |
| --- | --- |
| `./scripts/check_oss_readiness.sh` | Check public references and dead Python code |
| `./scripts/check_oss_readiness.sh --diff BASE COMMIT` | Check public references in changed files and dead Python code across the full configured source tree |
| `./scripts/check_public_references.py --diff BASE COMMIT` | Check public references only in files added or modified by the commit range |
| `./scripts/check_dead_code.sh` | Run the pinned Vulture scan across the full configured source tree |
| `./scripts/check_secrets.sh` | Manually scan the current tree for sensitive values before a release |
| `./scripts/check_secrets.sh --diff BASE COMMIT` | Scan only the commits reachable from COMMIT but not BASE, as CI does |
| `./scripts/check_public_references.py --strict` | List references to Kumo repositories that are not public yet |
| `./scripts/check_secrets.sh --history` | Perform the one-time full-history scan before a public release |
| `./scripts/check_secrets.sh --history-report /absolute/output/directory` | Perform that full-history scan and write a redacted review report |

Pull requests and pushes run `./scripts/check_secrets.sh --diff BASE COMMIT`
with a pinned, checksum-verified Gitleaks release from the OSS Readiness
workflow, scanning only the commits from the triggering event. To upgrade
Gitleaks, change the version in `.mise.toml` and the version and `linux_x64`
checksum in `.github/workflows/oss-readiness.yml` together. The other secret
commands remain for pre-release current-tree checks and the one-time
full-history release baseline.
