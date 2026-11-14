# Kumo examples

Notebooks and Python scripts for training, prediction and model reuse.

## Table of contents

- [Requirements](#requirements)
- [Kumo Engine](kumo-engine/README.md)
  - [Cross-industry](kumo-engine/README.md#cross-industry)
  - [Energy](kumo-engine/README.md#energy)
  - [Financial services](kumo-engine/README.md#financial-services)
  - [Healthcare](kumo-engine/README.md#healthcare)
  - [Hospitality](kumo-engine/README.md#hospitality)
  - [Media and entertainment](kumo-engine/README.md#media-and-entertainment)
  - [Retail](kumo-engine/README.md#retail)
  - [Science](kumo-engine/README.md#science)
- [Structured Data Models](structured-data-models/README.md)
  - [Relational](structured-data-models/README.md#relational)
    - [Cross-industry](structured-data-models/README.md#cross-industry)
    - [Financial services](structured-data-models/README.md#financial-services)
    - [Retail](structured-data-models/README.md#retail)
  - [Tabular](structured-data-models/README.md#tabular)
    - [Agriculture](structured-data-models/README.md#agriculture)
    - [Automotive](structured-data-models/README.md#automotive)
    - [Cross-industry](structured-data-models/README.md#cross-industry-1)
    - [Education](structured-data-models/README.md#education)
    - [Financial services](structured-data-models/README.md#financial-services-1)
    - [Healthcare](structured-data-models/README.md#healthcare)
    - [Hospitality](structured-data-models/README.md#hospitality)
    - [Real estate](structured-data-models/README.md#real-estate)
    - [Retail](structured-data-models/README.md#retail-1)
  - [Time series](structured-data-models/README.md#time-series)
    - [Cross-industry](structured-data-models/README.md#cross-industry-2)
    - [Retail](structured-data-models/README.md#retail-2)
- [Repository layout](#repository-layout)
- [Contributing](#contributing)

## Requirements

**There is no shared `requirements.txt` or dependency lockfile.** Run the install commands in your chosen notebook's setup cells or Python script's header; packages and extras vary by example.

| Workflow | Requirements |
| --- | --- |
| Local Kumo Engine | Linux and `kumo-engine` with the example's extras; the local Engine notebooks use CPU and writable local scratch storage. |
| Direct Structured Data Models | `structured-data-models` plus the example's dependencies; follow its Python and CPU/GPU requirements. |
| Client inference | `structured-data-models-client`, an existing model endpoint, and access credentials. |
| Databricks or Snowflake | The specified runtime, credentials and source-data access. Databricks setup cells install packages and restart Python. |

Follow each example's data preparation steps and dataset terms. Focused workflows may require an existing model or prepared dataset. Commented previews are not runnable demos.

## Repository layout

- `kumo-engine/<industry>/`: Engine examples using `kumo.engine`.
- `structured-data-models/<family>/<industry>/`: relational, tabular and time-series examples using `sdm` or `sdm_client`.

Each contains `notebooks/` and/or `sdk-code-examples/`; filenames identify the runtime. General workflows use `cross-industry/`. Copy companion directories with their examples.

## Contributing

Follow the [notebook guidelines](NOTEBOOK_GUIDELINES.md) and [focused module guide](MODULE_NOTEBOOK_GUIDELINES.md). Keep credentials and model outputs out of commits.

See the [source map](SOURCE_MAP.csv) and [attribution](ATTRIBUTION.md) for provenance.
