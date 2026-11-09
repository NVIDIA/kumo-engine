# G-GRAPH-001: graph sampling is bounded, point-in-time, deterministic and oriented

## Rule

A `NeighborhoodSpec` samples each sample's neighborhood layer by layer, starting
from the root entity. For every sample:

1. **Fanout and time.** Each layer samples at most `fanout` neighbors per
   frontier node for each `GraphSample`. When the traversal is temporal
   (`temporal="inherit"` with a destination clock), only neighbors dated at or
   before the anchor's cutoff are candidates. `strategy="recent"` keeps the
   `fanout` most recent candidates.
2. **Ties.** Which tied neighbors the engine keeps at the fanout limit is
   deterministic given the same data. It does not depend on row order, on how
   the tables are split into files, or on how an index is sharded. Which of the
   tied neighbors it keeps is otherwise unspecified.
3. **Edges.** An edge appears at most once per sample, however many times
   layer-wise expansion reaches it.
4. **Orientation.** Every edge of a relation points from its left table to its
   right table as the `RelationSpec` declares them, whatever the traversal
   direction or how the index stores it.

Nodes are unique per sample: a row reached more than once is one node.

## Boundaries

A neighbor dated exactly at the cutoff is a candidate; a later one is excluded. The cutoff
is each request's own, also when requests for the same entity are passed
together. A static traversal
(`temporal="static"`) is deliberately unfiltered; `strategy="recent"` still
orders its candidates by the clock.

Ties matter only at the fanout limit: when more candidates than the remaining
slots share the boundary timestamp. There, rule 1 fixes the timestamps of the
kept neighbors but not which of the tied neighbors they are. A sampled graph is
therefore checked layer by layer from the neighbors it kept: at every frontier
node, the kept neighbors are candidates and their timestamps are the `fanout`
most recent candidates' (included among them, when a relation is traversed more
than once and an edge's layer is ambiguous).

## Scope

The generated properties in `conformance/graph_cases.py` check it on the
DiskGraph engine: point in time, fanout bound and recency at every layer, node
uniqueness and orientation, edge uniqueness, determinism when sampling one index
again, and invariance to row order, files and shards. Every sampled graph is
checked layer by layer from the neighbors it kept, as described above, so the
check holds whichever tied neighbors the engine keeps. The DiskGraph engine
violates rule 2 (tied neighbors depend on row order and shards, and with 3
shards and 4 threads differ across fresh index builds of the same data); this is
recorded as a known gap in `conformance/registry.py` with a repro file, until it is fixed.
Edge uniqueness covers paths that traverse a relation again in a later layer:
repeated edges are dropped and the others keep their output order.

Uniform sampling, seeds and epochs, node features, sub-second clocks and timed
intermediate hops outside a neighborhood are outside this rule. The golden
format asserts raw entity values only, so no golden case holds a sampled graph.

## Golden examples

- [C-GRAPH-001](golden/C-GRAPH-001.json): the candidate set of rule 1, counted as
  an entity value over a relation declared opposite to the hop: neighbors dated
  at or before each cutoff, ties included. It exists because every guarantee
  needs a golden example; the properties above check the sampled graphs.
