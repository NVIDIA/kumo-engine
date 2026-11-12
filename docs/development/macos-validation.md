# macOS ARM64 desktop qualification

This suite runs the installed product on a native `macos-15` GitHub runner.
It covers durable save/restart/load/predict, supported retrieval dependencies,
and actionable errors for unsupported or incomplete configurations. Registration,
aliases, tracking, drift, and evaluation belong to the separate MLOps tests.

## Run

Open **Actions → Kumo Engine - macOS ARM64 qualification → Run workflow**.
Select the feature branch during PR validation or `main` after merging. The
Python selector defaults to **all**: 3.11, 3.12, 3.13, and 3.14. One native wheel
build supplies the exact same wheel bytes to all four jobs. No cloud credentials
or customer datasets are needed. Push events register the feature workflow but
skip execution.

The release platform matrix calls the same shared action. Each job installs
outside the checkout, verifies wheel/version/hash and installed module origins,
and checks dependencies before the basic smoke and desktop cases.

## Cases

| Case | Execution and required evidence |
| --- | --- |
| Durable CPU recovery | Reuse the small shared graph and training recipe. Train once; export the actual model and graph through public APIs; save complete reference predictions. Exit the producer and remove its source files and scratch. A different process loads only the retained exports into new scratch, predicts twice, and checks IDs, labels, finite probabilities, model/snapshot identity, and unchanged retained files. |
| MPS recovery | Repeat that process with explicit `accelerator="mps"`. Observe real MPS optimizer gradients and model-device selection. Missing MPS hardware is blocked; CPU fallback fails. |
| Published quickstart | Execute the README Python code against the packaged shop sample using explicit CPU execution. |
| Retrieval | Reuse actual context/two-tower prediction tests, top-K population/count checks, temporal item eligibility, custom prediction anchors, and native random-link exclusions. |
| Exports and continued training | Reuse existing real node/link export round trips, weighted training-table reimport, prediction/embedding exports, and warm-start tests that check restored parameters, optimizer state, refit modes, and source-bundle preservation. |
| Native sampling | Reuse native split-visibility, cutoff, and timestamp-unit assertions. |
| Incomplete model | Remove a required file from a copy of the exported model and require an actionable rejection. |
| Missing retrieval dependency | Actually uninstall Faiss from the disposable environment, start a fresh process, and require retrieval to identify the missing dependency and installation command. Restore the exact Faiss version and rerun the dependency check. |

Existing recipes run with explicit CPU configuration through the public `Trainer`
argument. The evidence plugin calls the real constructors and runtime; its session
hooks cover module-scoped training fixtures as well as individual tests. It does
not substitute training or prediction results. The independent MPS worker always
selects its device explicitly and disables PyTorch CPU fallback. Repeated and
reloaded probabilities use absolute tolerance `1e-4`, relative tolerance zero,
and exact IDs and labels; the receipt records this numerical policy.

These are focused desktop and related behavior cases, not a full learning-quality
campaign or a claim that every B1–B8 variant, GPU retrieval configuration, notebook
UI, or cloud platform is qualified. The recovery case uses the shared local
lifecycle's graph, model-plan, installation-audit, and process-management helpers;
it never invokes its MLOps phases.

## Results

Download `engine-wheel-cpu-macos-arm64-py<version>-*`. It contains installation and
dependency logs, `environment.json`, the basic smoke's JUnit result, and
`desktop/summary.json`. Desktop evidence includes each command, exit status,
runtime, JUnit counts, observed devices, worker receipts, wheel identity, and
recipe fingerprints. Resource values are process peak resident-memory measurements,
not simultaneous process-tree or GPU-memory maxima.

A skipped case, empty JUnit result, missing device evidence, failed worker, or
unavailable required MPS profile prevents a full pass. CPU and MPS profiles remain
separate in the summary. Failed attempts retain their own run IDs and artifacts;
a successful retry does not replace them. The 60-minute build deadline and
120-minute qualification deadline bound execution; individual cases also have
process-group timeouts.
