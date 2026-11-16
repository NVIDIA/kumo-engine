# Structured Data Models examples

Use direct models through `sdm`, or connect client examples through `sdm_client` to an existing model endpoint.
Choose a model family, then an industry and workflow below.
Client examples require an existing endpoint; they do not provision one.

| Model family | Examples |
| --- | --- |
| [Relational](#relational) | Direct and client KumoRelational workflows with linked tables. |
| [Tabular](#tabular) | Direct and client KumoTabular workflows, TabICLv2, fine-tuning and nearest-neighbor context. |
| [Time series](#time-series) | Commented forecasting and anomaly-detection previews. |

## Relational

### Cross-industry

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [RelBench entity binary/multiclass classification or regression](relational/cross-industry/sdk-code-examples/local_structured_data_models_relbench_prediction.py) | Local | Model workflow |

### Financial services

#### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Screen synthetic accounts with client KumoRelational](relational/fsi/notebooks/local_structured_data_models_client_money_muling_detection.ipynb) | Local | Model workflow |

### Retail

#### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Kumo Relational Quickstart with structured-data-models-client](relational/retail/notebooks/local_structured_data_models_client_relational_quickstart.ipynb) | Local | Model workflow |

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [SALT sales and item categorical field completion, evaluated with reciprocal rank](relational/retail/sdk-code-examples/local_structured_data_models_salt_relational_prediction.py) | Local | Model workflow |

## Tabular

### Agriculture

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Classify forest cover types](tabular/agriculture/sdk-code-examples/local_structured_data_models_forest_cover_type_prediction.py) | Local | Model workflow |
| [Classify leaf species](tabular/agriculture/sdk-code-examples/local_structured_data_models_leaf_classification.py) | Local | Model workflow |
| [Predict African soil properties](tabular/agriculture/sdk-code-examples/local_structured_data_models_afsis_soil_properties.py) | Local | Model workflow |

### Automotive

#### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [KumoTabular on Vehicle (OpenML 54) Walkthrough](tabular/automotive/notebooks/local_structured_data_models_vehicle_classification.ipynb) | Local | Model workflow |

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Classify ITK Fall 2018 records](tabular/automotive/sdk-code-examples/local_structured_data_models_itk_fall_2018.py) | Local | Model workflow |

### Cross-industry

#### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Explore TabICLv2 regression and classification](tabular/cross-industry/notebooks/colab_structured_data_models_tabicl_classification.ipynb) | Colab | Model workflow |

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Compare full, random and nearest-neighbor context](tabular/cross-industry/sdk-code-examples/local_structured_data_models_nearest_neighbor_context.py) | Local | Model workflow |
| [Fine-tune with nearest-neighbor context](tabular/cross-industry/sdk-code-examples/local_structured_data_models_nearest_neighbor_fine_tuning.py) | Local | Model workflow |
| [Full fine-tuning of KumoTabular ("kumo-small") or TabICLv2](tabular/cross-industry/sdk-code-examples/local_structured_data_models_tabular_fine_tuning.py) | Local | Model workflow |

### Education

#### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict passage readability with text features](tabular/education/notebooks/databricks_structured_data_models_text_readability.ipynb) | Databricks | Model workflow |

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict grant application outcomes](tabular/education/sdk-code-examples/local_structured_data_models_unimelb_grant_application.py) | Local | Model workflow |
| [Predict passage readability with text features](tabular/education/sdk-code-examples/local_structured_data_models_text_readability.py) | Local | Model workflow |

### Financial services

#### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict account closure with client KumoTabular](tabular/fsi/notebooks/local_structured_data_models_client_account_closure.ipynb) | Local | Model workflow |
| [Predict account closure with direct KumoTabular](tabular/fsi/notebooks/local_structured_data_models_account_closure.ipynb) | Local | Model workflow |

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict Santander customer transactions](tabular/fsi/sdk-code-examples/local_structured_data_models_santander_customer_transaction_prediction.py) | Local | Model workflow |
| [Predict Santander transaction values](tabular/fsi/sdk-code-examples/local_structured_data_models_santander_value_prediction_challenge.py) | Local | Model workflow |
| [Predict loan approval](tabular/fsi/sdk-code-examples/local_structured_data_models_playground_series_s4e10.py) | Local | Model workflow |

### Healthcare

#### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Kumo Tabular on Breast Mass Classification (Kaggle)](tabular/healthcare/notebooks/snowflake_structured_data_models_client_breast_mass_classification.ipynb) | Snowflake | Model workflow |
| [Kumo Tabular on Breast Mass Classification (UCI)](tabular/healthcare/notebooks/local_structured_data_models_breast_mass_classification.ipynb) | Local | Model workflow |

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Breast cancer classification and embedding capture](tabular/healthcare/sdk-code-examples/local_structured_data_models_breast_cancer_classification.py) | Local | Model workflow |
| [Identify age-related conditions with tabular classification](tabular/healthcare/sdk-code-examples/local_structured_data_models_icr_identify_age_related_conditions.py) | Local | Model workflow |
| [Predict biological responses](tabular/healthcare/sdk-code-examples/local_structured_data_models_bioresponse.py) | Local | Model workflow |

### Hospitality

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict restaurant revenue](tabular/hospitality/sdk-code-examples/local_structured_data_models_restaurant_revenue_prediction.py) | Local | Model workflow |

### Real estate

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict rental listing interest](tabular/real-estate/sdk-code-examples/local_structured_data_models_two_sigma_connect_rental_listing_inquiries.py) | Local | Model workflow |

### Retail

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Classify Otto Group products](tabular/retail/sdk-code-examples/local_structured_data_models_otto_group_product_classification_challenge.py) | Local | Model workflow |
| [Predict Mercari listing prices from product metadata and text](tabular/retail/sdk-code-examples/local_structured_data_models_mercari_price_suggestion_challenge.py) | Local | Model workflow |

## Time series

### Cross-industry

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Hourly forecasts, day-ahead energy prices and weekly retail sales](time-series/cross-industry/sdk-code-examples/local_structured_data_models_public_time_series_forecasting.py) | Local | Commented preview |
| [Multivariate synthetic time-series anomaly detection](time-series/cross-industry/sdk-code-examples/local_structured_data_models_time_series_anomaly_detection.py) | Local | Commented preview |

### Retail

#### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Forecast ice cream and cold drink demand with historical weather and known future prices](time-series/retail/sdk-code-examples/local_structured_data_models_weather_and_price_forecasting.py) | Local | Commented preview |

Commented previews preserve demonstrations of features unavailable in the supported package API.
Their commented demonstrations are not runnable.
Example-specific companion directories hold shared data preparation, helpers and assets; use them through their owning examples.
