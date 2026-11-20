# Dataset access

The private `kumo-engine-arena` bucket is in account `926922431314`, region
`us-west-2`. Repository access alone does not grant S3 access. The downloader
uses the AWS credential chain and pins the expected bucket owner.

The [catalog](inputs.json) now references one checksum-pinned inventory object.
CI has `GetObject` access to that exact object as well as the source/anchor grants
below. The downloader verifies the inventory before using its per-file pins;
it needs no bucket listing or writes.

For local use, select an existing organization profile with `--profile NAME`.
If that profile uses IAM Identity Center and its session has expired, sign in
with the organization's configured SSO process. Do not put credentials in the
suite or make the bucket public.

Kumo engine CI already uses GitHub OIDC with the
`gha-kumo-workflow-tests` role in the `gha-kumo-workflow-tests` environment.
The nightly AML job requests that role and performs the pinned downloads.
On 2026-11-16, CI assumed the role successfully but GetObject was denied for
all 11 HI-small inputs. With user authorization, a bucket-policy statement was
added granting that exact role `s3:GetObject` for only those 11 object ARNs.
The existing bucket owner restriction was preserved. The applied addition is
recorded in [ci-read-policy.json](ci-read-policy.json); its resources match the
HI-small entries in the pinned inventory. No IAM role policy was changed:
the available SSO identity cannot perform `iam:PutRolePolicy`.

The downloader does not list the bucket or write objects. The grant contains
no wildcard resources, other principals or write actions. Additional inputs
require a separately reviewed grant. The Engine role then downloaded and
verified all 11 objects in
[CI run 43857600971](https://github.com/NVIDIA/kumo-engine/actions/runs/43857600971):
375,509,265 transferred bytes in 2.84 seconds, including anchors and bindings.
This verifies CI access; it does not establish workload correctness.

## LI Medium access

With user authorization on 2026-11-17,
[ci-li-medium-read-policy.json](ci-li-medium-read-policy.json) was added for the
same Engine test role: `s3:GetObject` on ten exact LI Medium objects from
the pinned inventory. The eleventh input is the AML dataset manifest already covered by
the HI Small grant. The policy was reread and matched the approved addition;
both original statements, including the owner restriction, were unchanged.
No IAM role policy, list, write or wildcard access was added. Effective downloads
by the CI role still require verification in a LI Medium run.

## Synthetic Medium access

[ci-synthetic-medium-read-policy.json](ci-synthetic-medium-read-policy.json)
records separately approved read grants for the shared nine public-table prefixes,
source manifest, weekly-customer v1 anchors, weekly-account v1 anchors and LTV v2
anchors. The downloader verifies all selected objects against exact size and hash
pins and needs no bucket listing or writes. S04 adds only the LTV anchor Parquet
and manifest to the existing access; that two-object addition was applied and
verified by rereading the bucket policy. Effective CI-role downloads remain a
run-time check.

## MBD transaction access

With user authorization on 2026-11-18,
[ci-mbd-read-policy.json](ci-mbd-read-policy.json) was applied for the same Engine
test role. It permits GetObject for five exact client-membership objects, five
transaction partition prefixes (folds 0–4), and three exact manifest/anchor
objects. No listing or writes are granted. The downloader verifies 1,008 pinned
objects; fold -1 and other modalities are excluded. The policy was reread and
the approved addition matched; all six existing statements were preserved.
Effective CI-role downloads remain to be verified by the M04 run.

M03 adds two exact GetObject resources: the transaction-forecast v1 anchor Parquet
and manifest. This separately authorized addition was applied on 2026-11-18 and
verified by rereading the policy, preserving all seven prior statements. Its
1,008-object inventory shares 1,006 source/manifest objects with M04. Effective
CI-role downloads remain unverified locally; no full dataset was downloaded.
