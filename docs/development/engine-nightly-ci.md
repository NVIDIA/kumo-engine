# Nightly learning checks

The nightly case runner covers seven workload families on frozen 100M and 1B
inputs: adtech, delivery, commerce, streaming, community, payments, and fraud.
Twelve case variants run at each size. Each variant has a normal case and a
scrambled-control case. The complete roster contains forty-eight executions.

The [nightly runner guide](../../packages/kumo-engine/test/nightly/README.md)
is the maintained source for the four runtime records, training caps,
prediction samples, resource accounting, case commands, and collection.
Each case builds its own snapshot and training table. A complete collection
applies individual learning checks and paired normal/control checks.

Every reference creation and refresh generates fresh tables, answers, controls,
and prediction populations, then trains and predicts in the same measured case.
It validates current data and ignores older reference answers, counts, and hashes.
No separate answer-preparation job is required. Candidate mode still compares
generated stage identities with approved expectations and supports legacy payloads.
Source-only candidate controls need a pinned recipe and approved stage hashes.

Source-only bundles pin raw sources, graph metadata, descriptors, and generation
plans. They can omit old staging exports and answer or population payloads.
Each reference control needs the explicit seed-zero recipe described in the
[case behavior](../../packages/kumo-engine/test/nightly/README.md#case-behavior).
Admission rejects unsupported recipes before graph construction. Source bundles
still need preparation. Source-only metadata packages declare their raw objects
in a checksum-bound inventory. Do not declare missing answer payloads in the
archive or manifest.

Before training, reference mode saves exports and a `PREPARED` receipt at
`case/reference-data/reference-data.json`. The required `--reference-output-s3`
argument selects storage for proposed data. The runner uploads the prepared
answers and makes sure that the uploaded objects match their checksums before
fitting. A publication failure stops fitting. A model failure does not remove the published input evidence.
Publication never approves reference answers, scores, or resource requirements.

Each VM prepares only its assigned family at the selected scale through
`nightly_job.py prepare --case`. Source-only metadata packages bind raw S3 files.
Preparation downloads those files and computes SHA256 while streaming the download.
Measured case admission then reads each raw file once to make sure that its checksum
matches the manifest. No separate answer-generation job is needed.
Legacy family archives retain their declared payloads and parent manifests.
A missing family binding fails preparation without downloading every family.

Reuse immutable prepared inputs for cases in the same family and scale. Keep
snapshots, processes, scratch, accounting groups, and evidence separate for every
execution. Candidate qualification requires reviewed inputs, reference, policy,
and requirements, all bound by checksum. Reference measurements produce proposals
while numerical records await measurement and review.

Temporal-control import uses the public `allow_new_target_entities=True` option
so valid future-window labels survive import. The configured reference wheel
and candidate package must support this option. Reference version and checksum
come from `inputs.json`.

The host supplies a delegated cgroup parent before execution. The runner uses
a fresh child group before Engine import and graph construction. It measures
complete-case RAM and enforces a case deadline. The host guard resolves the
scratch mount to its physical disks and requires local NVMe instance storage.
EBS-backed or unresolved scratch paths fail before graph construction. This
storage check does not establish that a 25B case fits the available SSD.
The guide explains the [required host preparation](../../packages/kumo-engine/test/nightly/README.md#resources-and-preparation).

The shared case API supports the weekly integration contract. Weekly retains
cross-version model replay and covers 10B–25B in its separate change.
All seven workload families are part of this implementation at both
scales, including their controls. Instance scheduling, credentialed automation,
and automatic workload rotation remain separate work. The current collector
requires all forty-eight cases. This change adds no recurring GPU workflow.
