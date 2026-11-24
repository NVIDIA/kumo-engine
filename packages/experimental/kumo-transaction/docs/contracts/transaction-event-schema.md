# Transaction event schema

See `src/tfm/contracts/schema.py` (`TransactionEvent`, `EventLabel`).

Required: `entity_id`, `event_id`, `event_timestamp`. Timestamps are stored in
UTC. Categorical, numeric, and text maps are optional. Labels live in `labels`
and must not be duplicated as input fields.
