# Data configs

Configs that describe a dataset rather than a model. Recipe configs live in
each recipe's directory (`recipes/<slug>/config*.yaml`) and are complete on
their own; nothing here is merged into them.

- [`data/pragmatiq/world.yaml`](data/pragmatiq/world.yaml): the pragmatiq
  `WorldConfig` that generates the 100k-user, 31-month book the
  `pragma_like` and `blueprint_v2` reference configs are sized against. Its
  header gives the generate command.
  [datasets](../docs/reference/datasets.md) describes the book's role and split.
