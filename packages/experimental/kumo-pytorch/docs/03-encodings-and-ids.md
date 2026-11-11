# 3. Encodings and IDs

An encoder turns a typed sampled value into what the model reads. This page
covers three separate choices: which encoder a field uses and whether it's
shared with other fields, whether its IDs stay the encoder's own local IDs or
are translated into an ID space shared by several encoders (so they can use one
embedding table), and whether the model also gets an ID saying which field a
value came from.

## Encoders

Every field in a batch names its encoder explicitly; nothing is chosen from the
column's type or the input's container.

| Encoder | Accepts | Emits |
| --- | --- | --- |
| `Raw()` | Any scalar dtype | The value itself, plus validity |
| `ZScore()` | Integers, floats, decimals | Standardized floats (population statistics) |
| `Buckets(boundaries=...)` or `Buckets(bins=...)` | Integers, floats, decimals | Range IDs, from given cuts or learned quantiles |
| `Vocabulary()` or `Vocabulary(values=...)` | Any scalar dtype | One whole-value category ID |
| `Text(vocabulary_size=...)` | Strings | Variable-length byte-level BPE piece IDs |

A learned `Vocabulary`, `ZScore`, `Buckets(bins=...)` or `Text` whose every use reads a
stored column directly (an entity field, an event field or a graph node field)
learns from that table: from the rows whose clock is at or before the encoder
cutoff, each row counted once, whatever the training examples select. The
cutoff defaults to the latest anchor time the fitting split selects; pass
`encoder_cutoff=` to `Plan.materialize` to choose another time, or `None` to
read every row. A table without a clock contributes every row. The engine keeps
these column statistics with its data, so another task at the same cutoff
reuses them. `Buckets(bins=...)` cuts are `np.quantile`'s over the counted
values; past a million distinct values they come from exact log-spaced bins,
128 per doubling, within 0.6% of the exact quantile. `Text` trains its tokenizer on the column's
values streamed from the index in a fixed order, and the engine keeps the
tokenizer beside the statistics. Every other learned encoder (values computed
per anchor, such as aggregates and calculations, targets, and decimal or bytes
columns) fits on the training examples only.

A missing value is never zero or an unseen category: it keeps its own validity
bit (columns) or a reserved missing ID (categories). [The API reference](api/encodings.md)
lists every option.

## Private and shared encoders

An inline encoder is private to its one use. To share a fitted encoder, name it
in `BatchSpec.encodings` and refer to it with `UseEncoding`. Two encoders with
the same settings are still separate unless you name and share one.

Our example is a single users table with a currency column and a country column
that happen to share a value (`"USD"`), plus a balance.

```python
from kumo.pytorch import api

schema = api.DataSchema(
    tables=(
        api.TableSpec(
            name="users",
            primary_key="id",
            time_column=api.NON_TEMPORAL,
            columns={
                "id": api.DType(name="int64"),
                "currency": api.DType(name="string"),
                "country": api.DType(name="string"),
                "balance": api.DType(name="float64"),
            },
        ),
    )
)
sample = api.SampleSpec(
    entity=api.EntitySpec(table="users"),
    entity_values={
        "currency": api.Field(column="currency"),
        "country": api.Field(column="country"),
        "balance": api.Field(column="balance"),
    },
)
dataset = api.Dataset(
    tables={
        "users": api.InMemoryRows(
            rows=[
                {"id": 1, "currency": "USD", "country": "USD", "balance": 12.5},
                {"id": 2, "currency": "EUR", "country": "GB", "balance": 30.0},
            ]
        )
    }
)
anchors = (
    api.SampleRequest(sample_id="one", entity_id=1, as_of_time=0),
    api.SampleRequest(sample_id="two", entity_id=2, as_of_time=0),
)
```

The batch has three inputs. `currencies` is one named encoder used twice: once
with local IDs and once translated into the ID space `model`. The country
field has two private vocabularies, one per input.

```python
batch = api.BatchSpec(
    encodings={"currencies": api.Vocabulary()},
    id_spaces={
        "model": api.IdSpace(
            specials={"pad": "[PAD]", "unknown": "[UNK]", "missing": "[MISSING]", "mask": "[MASK]"}
        )
    },
    field_names={"names": api.FieldNames()},
    inputs={
        "local": api.Input(
            source=api.EntitySource(),
            representation=api.Fields(
                fields=(
                    api.Encode(field="currency", encoding=api.UseEncoding(name="currencies")),
                    api.Encode(field="balance", encoding=api.Raw()),
                )
            ),
            layout=api.Separate(),
        ),
        "tokens": api.Input(
            source=api.EntitySource(),
            representation=api.Fields(
                fields=(
                    api.Encode(
                        field="currency",
                        encoding=api.UseEncoding(name="currencies"),
                        id_space="model",
                    ),
                    api.Encode(field="country", encoding=api.Vocabulary(), id_space="model"),
                ),
                row=api.Tokens(),
                field_names="names",
                field_space="model",
            ),
            layout=api.Packed(),
        ),
        "country_again": api.Input(
            source=api.EntitySource(),
            representation=api.Fields(
                fields=(api.Encode(field="country", encoding=api.Vocabulary(), id_space="model"),),
                field_names="names",
                field_space="model",
            ),
            layout=api.Separate(),
        ),
    },
)
contract = api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)
```

## Inspecting fitted encoders

A task exposes its fitted encoders (`task.encoders`), ID spaces
(`task.spaces`) and, per output field, a read-only binding that encodes a value
exactly as that output does:

```python
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    api.prepare(contract=contract, dataset=dataset, engine=api.InMemoryEngine()) as plan,
    plan.materialize(anchors=anchors) as task,
):
    local = task.output(name="local").field(name="currency")
    translated = task.output(name="tokens").field(name="currency")
    country = task.output(name="tokens").field(name="country")
    independent_country = task.output(name="country_again").field(name="country")
    emitted = next(task.batches(batch_size=2))

    assert local.encoder is translated.encoder is task.encoders["currencies"]
    assert country.encoder is not independent_country.encoder
    print("USD, shared encoder, local ID:", local.encode(value="USD"))
    print("USD, shared encoder, final ID:", translated.encode(value="USD"))
    print("USD, private country encoder:", country.encode(value="USD"))
    print("USD, another private encoder:", independent_country.encode(value="USD"))
    print("special roles:", dict(task.spaces["model"].roles))

print("packed token IDs:", emitted.inputs["tokens"]["value_ids"].tolist())
print("sample boundaries:", emitted.inputs["tokens"]["offsets"].tolist())
print("field IDs:", emitted.inputs["tokens"]["field_ids"].tolist())
```

<!-- test: output -->
```text
USD, shared encoder, local ID: 3
USD, shared encoder, final ID: 11
USD, private country encoder: 9
USD, another private encoder: 7
special roles: {'pad': 0, 'unknown': 1, 'missing': 2, 'mask': 3}
packed token IDs: [11, 9, 10, 8]
sample boundaries: [0, 2, 4]
field IDs: [5, 4, 5, 4]
```

Here's what the numbers show:

- The shared encoder gives `"USD"` the same local ID in every use; joining the
  ID space translates it into a collision-free range. A use without
  `id_space` keeps local IDs even when another use joins a space.
- The two private country vocabularies get separate ranges in the ID space,
  so the same string has different IDs.
- Declared special IDs, such as `pad` or `mask`, have fixed numbers that don't
  depend on the data. Declaring one reserves its number; it shows up in the
  output only where padding, masking or history markers use it.
- Field-name IDs identify declared fields, not the letters of their names. A
  field keeps its field ID across inputs that use the same dictionary, even
  when its value encoders differ.

## Reusing fitted state

Everything a task fitted is in `task.fitted_state`. Passing it to another
`materialize(..., fitted=...)` reuses it as is: nothing is refitted and no ID
changes. `fitted_state.inputs` is the input half alone, for a
contract with the same inputs and different targets (fine-tuning on a new
label). See [training batches](05-training-batches.md#reusing-fitted-state) and
[inference and serving](06-inference-and-serving.md).

Next: [engines and data](04-engines-and-data.md).
