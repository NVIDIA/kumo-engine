# Third-party notices

The root `THIRD-PARTY.txt` is generated from `3rdparty/notices/inventory.json` and `3rdparty/notices/policy.json`. The inventory maps components copied into Kumo source or artifacts to the artifact that carries them. Dependencies installed as separate Python distributions normally retain their own legal files instead of being duplicated in the generated notice. A dependency is included when its installed distribution omits required upstream legal text.

Validate the inventory and committed notice:

```bash
python scripts/gen_third_party.py --check THIRD-PARTY.txt
python -m unittest test.scripts.test_gen_third_party
```

Validate the DiskGraph artifact manifest after populating Cargo's source cache:

```bash
cargo fetch --locked --manifest-path packages/diskgraph/Cargo.toml
python packages/diskgraph/scripts/generate_artifact_manifest.py --repo-root packages/diskgraph --check packages/diskgraph/third_party/artifact-manifest.json
```

After DiskGraph dependency changes, reconcile the inventory's Rust components with the artifact manifest and regenerate the notice in the same PR. Existing records are kept, crate versions that left both artifacts are removed, and new crate versions are derived from their root legal files in Cargo's source cache. Run these steps after populating the cache:

```bash
python scripts/refresh_diskgraph_notice_inventory.py
python scripts/gen_third_party.py --output THIRD-PARTY.txt
```

Review new crate records in the PR's inventory diff. A license outside the policy, a crate without license text and no reusable reviewed SPDX text, or a skipped vendored (path) crate fails the license notice check and needs a reviewed inventory or policy change.

After reviewing an inventory or policy change, regenerate the notice:

```bash
python scripts/gen_third_party.py --output THIRD-PARTY.txt
```

The `kumo-engine` wheel builder validates collected wheel identities, native members, and legal metadata before generating its artifact-specific notice. GloVe remains a separate dependency. See the [release guide](../docs/development/releasing-kumo-engine.md). Set `THIRD_PARTY_NOTICE` only to supply a reviewed alternative.
