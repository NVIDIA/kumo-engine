# G-COMPOSITE-001: One ID for an ordered scalar combination

## Rule

Composite encodes selected scalar components in declaration order and maps
their typed tuple to one outer vocabulary ID per eligible event. Ordinary tuple
IDs start at 2; missing and unknown remain distinct roles. Top-K retains the
most frequent tuples, breaking ties by canonical typed tuple order.

## Boundaries

A required NULL component removes that event. Optional NULL is a typed missing
component, not a placeholder string. Individually known components can form an
unseen tuple; unseen/top-K-excluded tuples use outer unknown ID 1. An unknown
component can participate in a known tuple. Raw and ZScore scalar components
are allowed. Text/repeated components refuse from the declaration even if
observed examples happen to contain one piece/occurrence. Components use local
results; only the outer ID joins an optional final dictionary. Component order
is semantic and is checked in supplied/reused state.

## Scope

The outer mapping is private to its input. This is one row arrangement using
the common scalar encoders, not an additional implicit per-field encoder.
Composite histories use G-HISTORY-001 assembly. Public field-name IDs are not
defined for a multi-field combination. General cross-input outer-map sharing
is outside this rule.

## Golden examples

- [C-COMPOSITE-001](golden/C-COMPOSITE-001.json): frequency/top-K, tuple order,
  optional NULL and required-row eligibility determine exact outer IDs.
