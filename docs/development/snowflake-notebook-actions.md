# Snowflake notebook engine checks in GitHub Actions

The trial workflow runs only on pushes to `egu/mlops-platform-integration` in
`NVIDIA/kumo-engine`. It does not run on `main`, `master`, or pull requests.
No default-branch change or draft PR is needed.

The selected check covers notebook preflight and the four engine lifecycle
phases: fit/register, fresh-process prediction and alias moves, monitoring and
late-label evaluation, and permission denial. It uses the identified installed
wheel. UI verification is excluded. Full epoch-history export remains a separate
unimplemented requirement; a successful engine check is not full qualification.
Both results are retained in `summary.json` and `ci-verdict.json`.

## GitHub setup

In repository **Settings → Environments**, create `snowflake-mlops-test`.
Under **Deployment branches and tags**, select **Selected branches and tags**
and allow only the branch `egu/mlops-platform-integration`.
The workflow checks this restriction before opening a deployment job; a missing
or differently restricted environment fails before any Snowflake connection.
A repository admin is required to create the environment. Add these environment
secrets:

- `SNOWFLAKE_PRIVATE_KEY`: the complete encrypted PEM private key, including
  its BEGIN/END lines, for `KUMO_SNOWFLAKE_CI`.
- `SNOWFLAKE_PRIVATE_KEY_PASSPHRASE`: its passphrase.

The local memory cache is not transferred to GitHub. The connector decrypts
these CI secrets in process; neither the key nor passphrase enters the notebook
payload or evidence. No browser credentials are needed for this check.

## Candidate and payload

[candidate.json](../../packages/kumo-engine/test/snowflake_notebook/candidate.json)
identifies the existing development wheel and dependency archive by SHA-256,
size, stage location, and observed object metadata. This trial tests that supplied
wheel; it does not build a wheel from the workflow commit. A future PR candidate
build must provide a corresponding manifest rather than reuse this candidate.

The runner submits the current checkout's test helpers. It checks that staged
assets retain their pinned metadata without downloading or uploading the large
archive. Inside the notebook, every referenced file is downloaded and checked
against its SHA-256 before installation or import. Offline pip resolves the
candidate's `snowflake` extra from the recorded dependency archive.

Changed or missing pinned assets fail rather than silently selecting another
wheel. Upload caching for the small helper files also uses content hashes.
Preflight runs before training. Attempts use unique model/experiment, stage
prefixes, and Code Bundles. Existing test resources are required; the runner
neither provisions infrastructure nor deletes retained evidence.

## Local validation and execution

Prepare and inspect without credentials or service calls:

```sh
python packages/kumo-engine/test/snowflake_notebook/prepare.py \
  --output-dir /tmp/snowflake-payload-001
python packages/kumo-engine/test/snowflake_notebook/launch.py \
  --payload-manifest /tmp/snowflake-payload-001/manifest.json \
  --preflight-payload /tmp/snowflake-payload-001/preflight \
  --lifecycle-payload /tmp/snowflake-payload-001/lifecycle \
  --output-dir /tmp/snowflake-result-001
```

For local RSA execution, unlock the key once with
`python packages/kumo-engine/test/snowflake_notebook/key_cache.py --key-file KEY`.
Use a new output directory, add `--key-file KEY --execute` to the launch command,
and keep the encrypted key outside Git. The Linux user-keyring cache expires
after eight hours and writes no decrypted key file. An expired/missing cache
fails without another implicit passphrase prompt.

The GitHub runner uses Python 3.12 and installs the Snowflake connector and
cryptography submission dependencies. Notebook dependencies are installed in a
fresh virtual environment. Each notebook has a 30-minute client deadline and
session SQL timeout with cancellation grace. Runs serialize; they do not cancel
an older run or automatically retry training. A runner interruption attempts
remote query cancellation, but a forced runner termination can still require
operator inspection using the saved execution query ID.

The Actions artifact retains receipts, hashes, query IDs, phase logs, the exact
payload manifest, and full qualification gaps. Review the reported wheel identity
and both engine/full-qualification verdicts. Missing secrets or access are
prerequisite failures, never passes.
