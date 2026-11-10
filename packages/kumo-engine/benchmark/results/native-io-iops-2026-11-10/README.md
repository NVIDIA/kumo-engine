# Portable IOPS evidence

This package contains 90 instrumented workflows. It preserves the original untraced latency experiment separately.

Run from the repository root:

```bash
python packages/kumo-engine/benchmark/native_io_iops_results.py --repo-root . --replay packages/kumo-engine/benchmark/results/native-io-iops-2026-11-10
```

Expected: `IOPS_REPLAY_PASS`, 90 records, six canonical variants and 18 backend groups. Source bytes and archive hashes must match. No historical graph/result paths, DiskGraph binary, live kernel, root or git object is required.

`rows/` uses the unchanged canonical row package; `raw-traces.jsonl.gz` binds exact trace and STOP texts to exact original rows. `observations.json` contains per-trial physical/native rates with separate clocks and medians of five rates. `iops-manifest.json` pins 10 measurement/replay source snapshots, the added helper SHA, and all archives. The report commit can descend from measured `ac7778cd134902178077d1ff179129e3ea40d91d`; replay requires pinned bytes, not that HEAD.

Each gzip archive is deterministic JSON Lines. Evidence records preserve original relative filename, exact text and SHA-256. The 20 stress trials remain HOLD (17.505% high-rate Direct observer inflation); the eight public controls passed their scoped n=2 E2E gate. Neither result establishes universal low overhead.

## Evidence inventory

- `stress20`: 243 exact text record(s), SHA-256 `9c2eae5ed394dafdc018150263d5d360395bd9269961958adc434f05e47cec92`.
- `public8`: 83 exact text record(s), SHA-256 `fc7af29201c6cb10301f7c31bd0b972192d09fd6a524bfdec9edb25bbd65b8ae`.
- `stress-driver`: 1 exact text record(s), SHA-256 `4d4a3b1c9eaf357c3734f5efd0ad248eb8f0236b5569d125291714e16a8742b7`.
- `stress-pair-driver`: 1 exact text record(s), SHA-256 `a7e7a950878b2043623d24b7e1963cfc787458e58601dc0c54a5e0c048081138`.
- `stress-reader`: 1 exact text record(s), SHA-256 `25b7098094fdb1bd0f373ca11ea3c810ade613e8f66ce6682d0effe3dfc2eae1`.
- `public-driver`: 1 exact text record(s), SHA-256 `3c12f1b6e886bd6792889011ec0307533fef624e18506eb22499e03a224c4fb5`.
- `public-reader`: 1 exact text record(s), SHA-256 `6023ea761291061320ef23029a1d4b6776ccf029c90d1f730dea2851d20ae3a8`.
- `overflow`: 13 exact text record(s), SHA-256 `029824f283fbdf178cd3d22fd97326e4e572acbefc3274256befa5ca6a68ef84`.
- `capability`: 1 exact text record(s), SHA-256 `38c3e6046f5bc7ce76ae9f03efecc9d566bb6a1c9390a8a07565a00c9b5b39c5`.
- `kernel-source`: 1 exact text record(s), SHA-256 `ebed7d78600350cd1edb6966e425928bd7ba744c8a6101d06b2b3d58d6c036ef`.
- `attribution`: 1 exact text record(s), SHA-256 `6d63fea7dd3cb9fbfe7921040e5c0dd4403d1e300360ebe52cca0328d24b4a71`.
- `collector-control-trace`: 1 exact text record(s), SHA-256 `c6bb4f845fb62a6404bbf8b2f5be0bf9bb304f6ea85c0a15d83a50dcbf251432`.
- `collector-control-result`: 1 exact text record(s), SHA-256 `a7c0e02982c3162385513563765f88e91efb53bf663ef4706e66c9faedcfaa5a`.
- `collector-control-stop`: 1 exact text record(s), SHA-256 `0f6284cf205c95e2e107f25b69470e8d6202874597202bc9c0f2b502542eab70`.
- `collector-control-script`: 1 exact text record(s), SHA-256 `b62c82d0493c93dd69e0e36d039f50d705b37c50cb63d6b6e5063d59d738ee94`.
- `build`: 1 exact text record(s), SHA-256 `13cd855acf1a4c40303a524bf9a1cb620cf679e329e0913f91f4e83c1a9cc463`.
- `launch-command`: 1 exact text record(s), SHA-256 `c77b5e3689bc3a3bf9949e399b43d05fa3319c13be2f21a6a335f8872e9d2379`.
- `executed-plan`: 1 exact text record(s), SHA-256 `eb758b64d2334f4eda093ae65076a2d3dc95af46f4f84f0f7e547191498b9977`.
- `progress`: 1 exact text record(s), SHA-256 `523d01436806f0c6e11a50b5706a9193149b29081113d3d48ad0b342c17db4b7`.
- `completion`: 1 exact text record(s), SHA-256 `f0a0bf4bda53c881b1796c44c999f2c1b8d87135ce4ea915387bf49a6bcc221d`.
- `final-verification`: 1 exact text record(s), SHA-256 `e8dbea74d2e1d8a046980fe9b9d4b636f007d8ff0f5bbe383a7e718a39cd8abf`.
- `launcher`: 1 exact text record(s), SHA-256 `7e76060d2442c91ec1d5a29a45b41594ca3a8ba38892a20721e326adc828ba46`.
- `final-verify-script`: 1 exact text record(s), SHA-256 `1f6b07121d0d07930ebc2e60d5abaa47e64fbc09775420e87190ce528332fba3`.
- `independent-final`: 1 exact text record(s), SHA-256 `2392684b7af9267347554753eec4b440458623a1a123d7a2dc83065d8a977526`.
- `independent-formal`: 1 exact text record(s), SHA-256 `47aee610574345f77722be19d646596573e897f8ee5c79fa3b2def6b5494900e`.
- `independent-formal-script`: 1 exact text record(s), SHA-256 `89c4a93fea48ef4e8e69b2155709c254fcec75a4a48e311525314f7030d14ef8`.
- `independent-public`: 1 exact text record(s), SHA-256 `8d78316a0cb1d6114f7e3b3ada201c5cc85b9a586a5a0aa52107b3a219940ac7`.
- `independent-public-script`: 1 exact text record(s), SHA-256 `bb86c3a4dde01ae95f09bb6c472505b1f685a7f70c7adc62a17fa4ad2d1989a0`.
- `independent-summary-script`: 1 exact text record(s), SHA-256 `c895f058460b6b307f1f20a12154be60f71dffaa9d0273e01fcdf6dbfe77b37c`.
- `original-completion`: 1 exact text record(s), SHA-256 `1e0de2dc0eba163130aa5fe9b115b631702d4fe7200d2902b16f95fabd7f4869`.
- `package-command`: 1 exact text record(s), SHA-256 `0c1882ed44e4e876cf369253cbe10c7d23a2bc157d464668728e9183bd4ed607`.
- `summary-binary30`: 1 exact text record(s), SHA-256 `0111d1a805a33c068f7b0ce1258cc4b923dab3ee9c32dfdd8c4b1185b5f4f6c6`.
- `summary-regression-resident`: 1 exact text record(s), SHA-256 `b97347e347919610f56417847d628ff60e71c57d3c912d5fa34dde77b7f594f8`.
- `summary-regression-constrained`: 1 exact text record(s), SHA-256 `11aa9e8cba084f18bcd6345c7db39f7718a1827b934a535b2da4bdc80ef20ba5`.
- `summary-temporal-link-resident`: 1 exact text record(s), SHA-256 `57e91f6bfadf41c85ec7a6a6c27a17bb6c27b35491485b5fd6a64efab697afd6`.
- `summary-temporal-link-constrained`: 1 exact text record(s), SHA-256 `7a67503fb29400d93ccda1a738360315747824780c3126d848c61de757ec9dd7`.
- `relocated-replay-proof`: 1 exact text record(s), SHA-256 `a5be6a31f279c57e45bb273a7a0da4841083e8a1f92ee81b8faa38193292686f`.
- `relocated-replay-script`: 1 exact text record(s), SHA-256 `61cb5c171baf05c7b421683c2167d7694e2f3b2574310bc49b5dfe2685a7d267`.
- `real-tamper-proof`: 1 exact text record(s), SHA-256 `5027462fc39cc539d8860de08d37a8b849bd8074e3fd0c999d526d5943ad5484`.
- `real-tamper-script`: 1 exact text record(s), SHA-256 `852cb415bd0317393185d02c8f6110325f2caca9ab1c746f6ea1b5f04dbdd094`.
- `independent-archive-review`: 1 exact text record(s), SHA-256 `ba530fcd487fb8654cfb11caff3107f7f23c016d855576a6cce239135cade7b6`.

Measurement source: `ac7778cd134902178077d1ff179129e3ea40d91d`. Native build source: `75ca0e2fd6797827feda61d637b59b75e2d6f678`. Reader SHA-256: `a17c4ad39e1e800367334e123ebfdbe346ce778cfc9122b8afa21751953f8b1b`.
