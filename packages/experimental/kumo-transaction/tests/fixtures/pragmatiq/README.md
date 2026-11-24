# pragmatiq synthetic fixture

Tiny synthetic parquet book matching the dynamiq-ai/pragmatiq on-disk contract.
No real customer data.

Regenerate:

```bash
uv run python -c "from pathlib import Path; from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture; generate_pragmatiq_fixture(Path('tests/fixtures/pragmatiq'), n_users=4, events_per_user=6, seed=0)"
```

`events_for_flat_fields` drops `app` rows; the key-value-time tokenizer keeps them.
