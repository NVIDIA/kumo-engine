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
Its first execution must verify both role assumption and object access; local
downloads do not establish that permission. If it fails, an administrator must
review the role/bucket permissions. This change applies no IAM policy.
The following identity-policy example is sufficient for this downloader's
GetObject requests; it is documentation, not an applied policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:GetObject",
    "Resource": [
      "arn:aws:s3:::kumo-engine-arena/datasets/aml-data/v8/*",
      "arn:aws:s3:::kumo-engine-arena/anchors/aml-data/v8/*"
    ]
  }]
}
```

The downloader does not list the bucket or write objects. Existing bucket
policy restrictions still apply; a policy allowing reads must actually be
attached to the caller. Permission errors fail acquisition rather than skip a
required dataset. Local download success does not verify the CI role.
