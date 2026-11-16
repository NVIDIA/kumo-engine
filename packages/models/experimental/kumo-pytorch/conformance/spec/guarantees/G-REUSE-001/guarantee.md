# G-REUSE-001: Frozen fitted state

## Rule

Compatible fitted state applies unchanged: numerical statistics, boundaries,
local vocabularies, field identities, final allocations, roles and output
bindings do not refit, grow or renumber. Held-out observations do not contribute
to training fit. Unknown later values use the saved encoder's declared fallback.

## Boundaries

Full-state reuse requires matching full-contract identity. Input-only reuse
requires matching input identity and may fit independent targets outside saved
spaces; it cannot add a member/role/allocation to a saved input-used space.
Saved input dependencies include complete final spaces needed to interpret
input IDs, even if a former target objective is no longer requested. Names
alone do not establish compatibility. On a fresh fit, repeated referring inputs
contribute their selected observations once per use: amounts 10 and 100 give
mean 55; an additional recent-only 100 view gives mean 70.

## Scope

This is the current in-process fitted-state protocol, not arbitrary fresh-process
artifact loading, version migration, registry discovery or configurable fitting
sources. A bound contract pins complete supported input dependencies; unrelated
target groups/thresholds can still need fitting. Lifecycle tests verify state
equality and forbid fitting entry-point calls on compatible reuse. The golden
below independently establishes the local mapping/fallback later reused.

## Golden examples

- [C-REUSE-001](golden/C-REUSE-001.json): supplied frozen currency entries
  map known values exactly and leave an unseen value at unknown ID 1.
