# kumo-glove-6b-50d

The glove.6B 50-dimensional word vectors that Kumo's text encoders use by
default, converted from Stanford's official `glove.6B.zip`
(https://nlp.stanford.edu/projects/glove/). GloVe is released under the Open
Data Commons Public Domain Dedication and License (PDDL 1.0).

Install it through the `text` extra of kumo-engine or kumo-core. The Kumo
Engine fat-wheel workflow builds it with `build_wheel.sh`, from the official
archive, checking the archive, the vocabulary and the vectors against pinned
SHA-256 digests.
