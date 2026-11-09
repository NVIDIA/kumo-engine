# G-IDSPACE-001: Shared encoders and final dictionaries

## Rule

A named encoding reference shares one fitted transform. An inline encoding is
private to its declaration, even when two declarations have identical settings
or reuse the same Python object. A use without id_space returns local IDs; a
use naming a final IdSpace translates IDs through that dictionary. Destination
membership belongs to the use and does not propagate to its other uses.

## Boundaries

Shared encoders require identical resolved operand types. Sharing a final space
does not pool independent encoders' observations or collapse their ordinary
entries. Reserved roles allocate in pad, unknown, missing, mask, entity, event,
separator, bos, eos order, followed by custom names in lexical order. Members
then allocate in canonical structured-identity order and local-ID order.
Applicable local missing/unknown/pad entries map to their declared final roles;
ordinary entries never collide merely because their local numbers match.
Equal display strings do not merge roles, field identities, and data values.

## Scope

FieldNames identifies declared sampled fields, not words in their names. Its
members come from declarations, not observed rows. Two uses of the same sampled
field in one explicitly shared field dictionary reuse its identity; different
event variants remain distinct qualified identities. Field maps reserve padding
when needed; a structural marker has no field owner. Resource-map order does
not alter allocations. Changing a fitted contract may change numbering;
G-REUSE-001 protects unchanged mappings under compatible reuse.

## Golden examples

- [C-IDSPACE-001](golden/C-IDSPACE-001.json): three inputs demonstrate shared
  currency fitting, private amount/merchant mappings, optional local output,
  shared field identities, reserved roles, and exact final IDs.
