# Dataset access

The private `kumo-engine-arena` bucket is in account `926922431314`, region
`us-west-2`. Repository access alone does not grant S3 access. The downloader
uses the AWS credential chain and pins the expected bucket owner.

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
HI-small entries in [inputs.json](inputs.json). No IAM role policy was changed:
the available SSO identity cannot perform `iam:PutRolePolicy`.

The downloader does not list the bucket or write objects. The grant contains
no wildcard resources, other principals or write actions. Additional inputs
require a separately reviewed grant. CI must verify actual downloads after the
permission change; owner-profile access alone is not evidence of CI access.
