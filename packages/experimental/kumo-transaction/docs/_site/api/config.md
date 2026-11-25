# `tfm.config`

Every recipe YAML validates against `RecipeConfig`. Unknown or removed keys
raise; the [config schema](../../reference/config-schema.md) explains each
section in prose.

::: tfm.config
    options:
      members:
        - RecipeConfig
        - load_recipe_config
