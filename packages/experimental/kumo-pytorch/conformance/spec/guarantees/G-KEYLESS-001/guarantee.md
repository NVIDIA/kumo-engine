# G-KEYLESS-001: keyless tables preserve source occurrences

## Rule

A table may declare `primary_key=None`. Each source row is a separate
occurrence, even when all its values equal another row's values. Traversal,
aggregates, event sequences, row targets, and graph sampling preserve those
occurrences. `Rows(distinct=True)` removes repeated visits to one occurrence;
it does not merge separate equal-valued source rows. Repeated graph visits
reuse the occurrence's node, and each edge appears once per sample.

An entity table must declare a primary key. This includes sample roots and
candidate entities. A keyless entity is refused during validation, before any
source access. Relations still join their declared columns; DiskGraph retains
its existing requirement that exactly one relation side be a declared PK.

## Boundaries

Historical windows retain `start < time <= stop`, including cutoff equality.
Empty histories have count and sum zero. Separate identical eligible rows each
contribute to counts and sums. A repeated visit contributes again only when
`distinct=False`.

Internal row/node identities are opaque and local to the prepared dataset or
index. They are not source columns, entity IDs, or stable identifiers across
reordering, file repartitioning, shards, or index rebuilds. Equal-time rows in
a keyless table use engine-defined internal identity for tie breaking; no
cross-engine or cross-build order of tied rows is promised. Existing keyed
identity, ordering, validation, and source-row cleanup behavior is unchanged.
Serving subsets also have their own local identities and source order. Opaque
graph IDs are compared under renaming. Equal-time ordering, tied limits,
uniform draws, and floating reduction rounding can differ across preparations
or engines. Counts and integer sums remain exact. A declared PK is needed for
stable cross-preparation choices; see [G-SERVING-001](../G-SERVING-001/guarantee.md).

## Scope

DiskGraph supports keyless tables within its existing operation limits. The
in-memory test engine also executes these cases for differential comparison.
Serving accepts separate identical keyless rows and only requires PK columns
and key uniqueness for keyed tables. Existing limits on empty native tables,
join shapes, intermediate clocks, and other operations remain in force.

The goldens assert raw aggregates calculated below. Focused tests additionally
check graphs, event delivery on both output formats, entity refusals, schema
inference, artifact reuse, and serving. Graph comparisons assert occurrence
multiplicity, features, and connectivity rather than literal internal IDs.

## Golden examples

- [C-KEYLESS-001](golden/C-KEYLESS-001.json): identical source rows, cutoff
  equality, a bounded recent window, future exclusion, and an empty history.
- [C-KEYLESS-002](golden/C-KEYLESS-002.json): two static paths revisit two
  identical facts; distinct counts two and non-distinct counts four.
