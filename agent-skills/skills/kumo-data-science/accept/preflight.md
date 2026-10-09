# Read-only Preflight — inferring structure before the graph gate

Once the required fields are present, the agent resolves **structure** itself. It does not ask
the user for column names, keys, or joins, and it does not call Kumo at this point — preflight is
catalog SQL against `INFORMATION_SCHEMA` plus explicitly targeted, read-only probes against
source data, and it is read-only throughout. A probe exists only to produce evidence for the
proposal; it is never written anywhere and never becomes a model input.

Preflight produces the **graph proposal** the user confirms at Node 1c. Every value carries a tag:

| Tag | Meaning |
|---|---|
| `[supplied]` | the user stated it; never overridden |
| `[declared]` | read from a semantic view's definition (§1a); shown for confirmation |
| `[inferred]` | measured by preflight; shown for confirmation |
| `[inferred, unverified]` | a candidate preflight could not establish, only fail to disqualify; must be confirmed before use |
| `[default]` | a contract default (`table_mode: all-connected`); shown for confirmation |

Precedence: `[supplied]` > `[declared]` > `[inferred]`. `[inferred, unverified]` is not a rung
on that ladder but a weaker form of `[inferred]`: whatever its source, a value carrying it must
be confirmed before the table can be an entity or an edge destination.

If the harness or host has already run preflight and handed you a result (for example a
"Preflight result for <schema>" message), use that result as-is — do not ask the user to repeat it.

---

## 1. Resolve `source_scope`

```sql
SELECT SCHEMA_NAME FROM <DB>.INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME ILIKE '<schema>';
SELECT TABLE_NAME, ROW_COUNT FROM <DB>.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA ILIKE '<schema>';
```

- Not found, or zero tables → **ASK BACK**: the source cannot be resolved; ask for a valid
  database and schema (or connector). Do not guess a similarly named schema.
- Found → enumerate tables **only inside this scope**. Never search other schemas, and never
  rank or curate tables by business relevance.

## 1a. `source_scope` is a semantic view

A semantic view is not a schema to search. It is a definition that points at physical tables and
declares their keys and relationships. Read the definition; do not query the view itself.

```sql
DESCRIBE SEMANTIC VIEW <DB>.<schema>.<view>;
```

The result holds one `property` / `property_value` pair per object, not one column per field.
Group the rows by `object_name` for each `TABLE` and `RELATIONSHIP` object, then read each field
as "the row whose `property` is X, taking `property_value`": for a `TABLE` object read
`BASE_TABLE_DATABASE_NAME`, `BASE_TABLE_SCHEMA_NAME`, `BASE_TABLE_NAME` and `PRIMARY_KEY` this way;
for a `RELATIONSHIP` object read `TABLE`, `REF_TABLE`, `FOREIGN_KEY` and `REF_KEY` this way —
`TABLE` and `REF_TABLE` are the view's logical names too; map each to its physical table through
the extracted `TABLE` records before using it anywhere downstream.

- Not found or unreadable → **ASK BACK**, exactly as for a schema that does not resolve.
- **A logical table with no `BASE_TABLE_*` properties does not name a physical table.** One
  defined by a query carries its own definition instead. The SDK can read a query through
  `connector.sql(...)`, but this skill does not build graph tables from SQL it composed itself, so
  the view alone is not enough. **ASK BACK**: name the logical table, say it is defined by a query
  rather than a base table, and ask for a view whose tables are physical or for the physical table
  to use. Do not parse the definition or rebuild it as a query of your own.
- **Only a plain column-to-column equality relationship becomes an edge.** A Kumo `Edge` is
  foreign key to primary key, matched on equality. If a relationship declares any other kind of
  match, do not read its `FOREIGN_KEY` / `REF_KEY` as an ordinary key pair: a non-equality payload
  can read as a one-column key and become an edge that does not mean what the view says.
  **ASK BACK**: name the relationship and its kind, and say only equality joins become edges.
- **The base tables must all sit in one database and schema.** The connector reads from a single
  schema. If they span more than one, **ASK BACK**: name the schemas found and ask for a semantic
  view or a schema whose tables share one schema. Do not pick one schema, and do not drop tables
  to make it fit.
- That one schema becomes the effective scope: it is the `database` / `schema_name` passed to
  the connector, and §2 reads columns from it.
- **The tables in scope are exactly the view's base tables** — not every table in that schema.
  The user chose them by naming the view; that is their selection, not curation by the agent.
- **Always use physical names.** The view's logical names (`USERS`, `TRANSACTIONS`) do not exist
  for Kumo. Every table, key, link, edge and PQL reference uses the base table and column names,
  with the same case and quoting the catalog reports.
- Each `PRIMARY_KEY` and each relationship is `[declared]`. They replace §3 and §5 for the tables
  they cover; §3 and §5 still apply to any base table the view gives no key or link for.
- A key with more than one column, on either side of a relationship, cannot become a Kumo edge.
  **ASK BACK**: name the relationship and say a single-column key is needed, the same way as a
  view that does not resolve — ask for a corrected view or schema. Do not offer to drop the
  table, concatenate the columns, or substitute a different link as options. Those are repairs,
  not a corrected source, and a menu of repairs is not an ask-back.
- A declared `PRIMARY_KEY` with more than one column only matters when it is what some
  relationship's `REF_KEY` points **at**. If no relationship's `REF_KEY` is this composite key,
  it needs no ask-back by itself — not even when this table is the **foreign-key** side of its
  own relationships and one of those `FOREIGN_KEY` columns happens to also be part of the
  composite key; pointing out of a table never depends on that table's own primary key. Leave the
  table in the graph exactly as §3 treats a table with no usable primary key: keyless, or with a
  single-column `[inferred]` key (e.g. an `*_ID` column) offered for confirmation. That keyless or
  `[inferred]` treatment applies only while **no** relationship references this table at all. If
  some relationship's `REF_KEY` is a different single column of it, that column is the key, by the
  bullet below; do not infer another, because an edge points at the key Kumo is given and a guessed
  one joins the foreign key to the wrong column. Ask back only when some relationship's `REF_KEY`
  is the composite key — that is the previous bullet, not this one.
- A relationship whose **single-column** `REF_KEY` is not the referenced table's declared
  `PRIMARY_KEY` (for example it references a `UNIQUE` key instead) is still expressible. A Kumo
  primary key is declared by the caller in `Table.from_source_table(..., primary_key=)` and does
  not have to match the catalog, so that column becomes the table's Kumo primary key, shown
  `[declared]`. Say in the proposal that it differs from the catalog's primary key so the user can
  correct it. This is not a repair; it is reading the view as written.
- That holds only while one column can serve the whole table. When **two relationships reference
  two different columns of the same table**, a Kumo table has one primary key and cannot carry
  both. **ASK BACK**: name both relationships and the column each references, say only one can be
  an edge destination, and ask for a corrected view, or for explicit authorization to build without
  one named relationship. Asking only which column to key on is not enough, because either answer
  still leaves the other relationship unbuildable. Do not silently key on one and leave the
  other relationship unbuilt, and do not emit the table twice to satisfy both: separate copies are
  not joined to each other, so `A -> Entity <- B` would silently become two disconnected
  components, and an entity split this way is ambiguous as the `FOR EACH` target.
- Ignore the view's dimensions, facts and metrics. They describe aggregate reporting, not the
  graph. In particular, a metric never supplies `prediction_target` or a positive-label condition
  — those still come from the user (`intake_contract.md`).

## 2. Columns and types

```sql
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM <DB>.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA ILIKE '<schema>'
ORDER BY TABLE_NAME, ORDINAL_POSITION;
```

## 3. Primary key per table

In order of preference, first match wins:
1. A declared primary key (`SHOW PRIMARY KEYS IN SCHEMA <DB>.<schema>`).
2. `<TABLE_STEM>_ID`, where the stem is the table name without a `_DIM` / `_PROFILE` / `_EVENTS`
   suffix, singular or plural (`CUSTOMER_ID` in `CUSTOMERS`, `PAGE_VIEW_ID` in `PAGE_VIEW_EVENTS`).
3. A column named `ROW_ID`, `ID`, or `EVENT_ID`.
4. For dimension / profile tables only: `<TOKEN>_ID` for any token of the table name
   (`ACCOUNT_ID` in `ACCOUNT_PROFILE`). Never for event tables — there such a column is a link.
5. A column that one or more *other* tables in scope carry under the same name is a **candidate**
   only. Shared naming does not make it a key, and neither does uniqueness: `COUNTRY`, `STATUS` or
   `MERCHANT` can be ordinary attributes, and an accidentally unique attribute is still not an
   identifier. A duplicated inferred key silently discards rows. A column that rules 1-4 already
   established as another in-scope table's key is a link to that table, not a candidate here —
   skip it, as rule 4 does for event tables.

   Use a targeted read-only probe to **reject** candidates, never to establish identity:

   ```sql
   SELECT COUNT(*) AS N, COUNT(DISTINCT <COL>) AS N_DISTINCT, COUNT(<COL>) AS N_NON_NULL
   FROM <DB>.<SCHEMA>.<TABLE>;
   ```

   Anything other than `N > 0 AND N_DISTINCT = N_NON_NULL = N` disqualifies the candidate outright
   — an empty table returns three zeros and does not qualify. Surviving the probe makes it a
   **proposed** key, tagged `[inferred, unverified]`, which the user confirms or corrects at the
   gate before it is used as the `FOR EACH` entity or an edge destination. If several candidates
   survive on one table, or the same candidate survives on more than one side, report the
   ambiguity and propose none. If the probe cannot run, list the column under `Not found` as an
   unverified candidate.

6. Otherwise preflight **inferred no primary key**. Report that as its own line and treat the
   table as keyless unless corrected. A keyless table may be an edge *source*, but not an
   `Edge.dst_table` and not the `FOR EACH` entity. Do not conclude from this failure alone that
   the table is an event or fact table — it may have a natural, composite or unconventional key
   the heuristic missed. Never promote a merely unique-looking column to primary key: a column
   unique in the rows you can see is not a declared key, and a child table may hold several rows
   per parent.

Rules 1-4 establish a key and name it `[inferred]`. Rule 5 never establishes one: it proposes a
survivor as `[inferred, unverified]`. Until the user confirms it, the table is treated exactly as
keyless — usable as an edge source, but not as an edge destination and not as the `FOR EACH`
entity. Confirmation is what lifts that restriction, not the existence of a candidate.

## 4. Time column per table

The first `TIMESTAMP*`, `DATE`, or `DATETIME` column, preferring names ending in `_AT`, `_TS`,
`_TIME`, `_DATE`, or named `EVENT_TIME` / `TIMESTAMP`. A table with none is static.

For every time-ranged target aggregation, the table it references must have a usable
`time_column`. If it does not, say so in the proposal and say what would make the task
expressible: a timestamp on some other table does not repair the target.

A catalog `TEXT` column may still be usable when its date semantics are established and the
generated code casts it with `dtype=Dtype.date`; the absence of a timestamp-typed catalog column
is not by itself grounds for rejection. Integers that merely order or describe events — a
sequence number, a day-of-week, an hour, a gap in days since a previous event — are not
timestamps and must not be treated as one.

## 5. Column availability

For each table, place every candidate feature in time relative to that table's proposed
`time_column`, then exclude the ones that are not yet knowable at it.

- When the time column marks the beginning of a validity period or interval, exclude a feature
  whose value is only determined once that interval later changes or ends: its eventual endpoint,
  its realized duration, or any summary of the completed interval.
- Exclude a feature computed from, or containing, the confirmed prediction target.
- Apply the active host adapter's runtime column constraints.

Decide by semantic role, never by a fixed column-name list. Record every exclusion and its reason
in the proposal. Where the role is genuinely ambiguous, do not invent one — show the column as an
exclusion the user can overturn at the gate, rather than asking the user for column names.

## 6. Links

A link is proposed when a column in table A has the **same name** as the primary key of table B
(`FRAUD_EVENTS.CUSTOMER_ID → CUSTOMERS.CUSTOMER_ID`). Foreign-key constraints, if declared, are
taken as well (`SHOW IMPORTED KEYS IN SCHEMA ...`). Nothing else is inferred — no fuzzy matching,
no value overlap checks, no joins executed.

## 7. Seeds and table mode

- **Seeds** are the tables behind `prediction_entity` and `prediction_target` (for link
  prediction: source entity, item entity, and the interaction table). Resolve them by name
  against the enumerated tables; if none matches, ask the user which table holds the entity or
  the outcome — that is intent, not structure.
- **Preflight runs only after Node 1a passes.** It resolves tables for fields the user already
  supplied; it never back-fills a field the user left out. If the required-fields check found
  `positive_interaction`, `recommended_item`, `positive_label_condition` or `time_window`
  missing, ask for them first — do not read them off the schema.
- **`table_mode: all-connected [default]`** — every table reachable from the seeds through the
  proposed links is included. List any table that is **not** reachable so the user can drop it
  or name a link.
- If the seeds themselves cannot be connected through inferred links, say so and **generate
  nothing** — connecting them would require creating or transforming data.

## 8. The graph proposal

Present exactly this shape, then **stop and wait** for confirmation:

```
Source scope: USER_DATA.E_COMMERCE [supplied] — 7 tables found
Table mode: all-connected [default] — 7 of 7 tables reachable from the seeds

Metadata
- customers: primary key CUSTOMER_ID [inferred]
- fraud_events: primary key EVENT_ID [inferred]
- fraud_events: time column EVENT_AT [inferred]
- ...

Links
- fraud_events.CUSTOMER_ID -> customers.CUSTOMER_ID [inferred]
- ...

Excluded columns
- fraud_events.RESOLVED_AT [inferred] — not knowable at the row's EVENT_AT
- fraud_events.DAYS_TO_RESOLVE [inferred] — duration of an interval that has not ended yet
- ...

Not found
- payments: no primary key inferred [inferred] — treated as keyless; edge source only unless corrected
- events: no usable time column [inferred] — a time-ranged target cannot reference this table
- ...

Confirm, or correct any line (your values take precedence).
```

When `source_scope` is a semantic view, the first line names the view and the schema it resolved
to, and keys and links read from it carry `[declared]`:

```
Source scope: semantic view SALES_DB.ML.CHURN_SV [supplied] — 3 base tables in SALES_DB.ML
...
- orders: primary key order_id [declared]
- orders.customer_id -> customer_profile.customer_id [declared]
```

`Excluded columns` and `Not found` are local availability-gate extensions to the confirmation
format.

## 8. Never

## 9. Never

- Prepare, transform, aggregate or materialize **model inputs** — joining, cleaning,
  deduplicating, creating labels. A diagnostic scalar aggregate over source rows is permitted
  solely as evidence for the proposal: row, distinct and null counts for a candidate key. Its
  result is reported and then discarded; it is never written anywhere and never reaches training.
- Rank tables by relevance or drop tables the user did not ask to drop.
- Proceed to PQL before the proposal is confirmed.
- Ask the user for columns, types, keys or joins. If preflight cannot run (no SQL access), say
  that plainly and ask the user to confirm the proposal you can build from what they said —
  still tagged, still confirmed, still no invented values.
