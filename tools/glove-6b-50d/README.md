# kumo-glove-6b-50d

The glove.6B 50-dimensional word vectors that Kumo's text encoders use by default, converted from Stanford's official [`glove.6B.zip`](https://nlp.stanford.edu/projects/glove/). The Python wrapper is licensed under Apache-2.0; the pretrained vectors are released under the Open Data Commons Public Domain Dedication and License (PDDL 1.0). The wheel includes `LICENSE` and `THIRD-PARTY.txt` with the applicable license texts and attribution.

Install it through the `text` extra of kumo-engine or kumo-core. The Kumo Engine fat-wheel workflow builds it with `build_wheel.sh`, from the official archive, checking the archive, the vocabulary and the vectors against pinned SHA-256 digests.

Build a complete wheel with `build_wheel.sh OUT_DIR` from a Git checkout. The script checks the scoped notice and completed wheel's legal files before copying output. Supplying `PYTHON` uses that interpreter's prepared dependencies; the default build uses the script's pinned toolchain. Licensing CI builds from the checked-in sources to inspect legal metadata without downloading the model; use the full builder before distributing the wheel.

Regenerate the scoped notice from the shared inventory with:

```bash
python scripts/gen_third_party.py --artifact glove-6b-50d-bundled-data --output tools/glove-6b-50d/THIRD-PARTY.txt
```
