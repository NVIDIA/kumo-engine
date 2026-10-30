# Engine nightly check

One job uses the existing `linux-amd64-gpu-t4-latest-1` runner and GPU0.
It builds and installs the selected wheel, runs sixteen cases in order,
runs required behavior checks, compares results, and uploads artifacts.
Each of eight tasks uses seed zero with normal and scrambled answers.
Keep the frozen PR172 `100M` data and complete training recipes.

Require actual CUDA work, complete finite predictions, exact saved state,
reload checks, native input rules, and valid filtered rankings.
Missing or failed checks fail the job. Quality and unqualified numerical
differences report only. Count extra behavior fits separately.

## Inputs and reference

The [input descriptor](../../packages/kumo-engine/test/nightly/inputs.json)
pins the private S3 archive, input manifest, and reference checksums.
The job downloads data into temporary storage through existing AWS access.
AWS download tools stay outside the measured Engine environment.
The approved wheel requires Lightning 2.6.5; the candidate requires 2.6.6.
Pin each complete 98-package runtime and report this dependency difference.

The initial descriptor uses the RTX recipe predecessor for setup only.
It cannot pass an ordinary T4 comparison.
Collect the initial T4 reference with manual input `reference_only: true`.
This installs the exact approved `kumo-engine==1.0.0` wheel and runs the same
sixteen cases and behavior checks. It exports a verified reference index.
Admit that index in immutable storage and update the descriptor checksum
before running ordinary candidate comparisons. Keep RTX results historical.
They cannot qualify the T4 reference. Ordinary runs reuse the saved T4
reference and train only sixteen fresh primary candidate models.

Reference and candidate hardware profiles must match, including driver,
Torch/CUDA, GPU model and memory, architecture, and thread settings.
Record each runner's GPU UUID without requiring the same physical device.

Record operation times, sampled process-tree RAM, device GPU memory,
and Torch allocations. PSS counts each process's share of shared RAM.
Summed RSS can count shared RAM more than once.
Record graph preparation, training, predictions, and model loading separately.
The engine total excludes build, installation, and separate behavior checks.
Report comparable changes against the T4 reference. Missing measurements stay
unavailable. These measurements set no performance limits.

## Schedule and capacity

Run daily at 10:00 UTC, manually with `product_ref`, or through approved
copied PR branches. Normal NVIDIA workflow vetting still applies.
The T4 catalog lists 6 CPUs, 24G RAM, and 150G SSD. Full-workload capacity
and complete runtime remain unverified. Keep inputs and work unchanged if
capacity fails. This workflow creates no machines and publishes no packages.
