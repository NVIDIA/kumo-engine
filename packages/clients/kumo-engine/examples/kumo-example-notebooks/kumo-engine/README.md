# Kumo Engine examples

Train relational models, prepare datasets and reuse models through `kumo.engine`.
Choose the vertical and workflow below, then follow that example's setup and data prerequisites.
For direct or client KumoRelational inference, see the [Structured Data Models relational examples](../structured-data-models/README.md#relational).

The synthetic industry notebooks below generate their own related tables and use the local Engine API through training, batch scoring on newly added records, and model reload. They require a prepared Linux GPU environment and local SSD scratch.

## Cross-industry

### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Build a custom training table](cross-industry/notebooks/local_kumo_engine_custom_training_table.ipynb) | Local | Focused workflow |
| [Choose a static or temporal query](cross-industry/notebooks/local_kumo_engine_static_and_temporal_queries.ipynb) | Local | Focused workflow |
| [Choose the customers to score](cross-industry/notebooks/databricks_kumo_engine_custom_prediction_population.ipynb) | Databricks | Focused workflow |
| [Preview graph visualization](cross-industry/notebooks/local_kumo_engine_graph_visualization_preview.ipynb) | Local | Commented preview |

### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Compare a retrained challenger with a trusted champion on identical held-out rows](cross-industry/sdk-code-examples/local_kumo_engine_champion_challenger.py) | Local | Model workflow |
| [Compare neighbor sampling choices on the same CustomerLTV training rows](cross-industry/sdk-code-examples/local_kumo_engine_model_iteration.py) | Local | Model workflow |
| [Configure delimited strings and native arrays as categorical collections](cross-industry/sdk-code-examples/local_kumo_engine_multicategorical_columns.py) | Local | Focused workflow |
| [Fit a users repeat-purchase model and retain its exact bundle for a score job](cross-industry/sdk-code-examples/local_kumo_engine_mlops_fit.py) | Local | Model workflow |
| [Fit and retain a CustomerLTV model for retraining in a fresh Python process](cross-industry/sdk-code-examples/local_kumo_engine_train_reusable_model.py) | Local | Model workflow |
| [Inspect a Snowflake schema with a configured local connector identity](cross-industry/sdk-code-examples/snowflake_kumo_engine_snowflake_connectors.py) | Snowflake | Focused workflow |
| [Inspect rooted and any-URI Amazon S3 connectors using ambient AWS credentials](cross-industry/sdk-code-examples/local_kumo_engine_s3_connectors.py) | Local | Focused workflow |
| [Preview interactive inspection of a small synthetic shop graph](cross-industry/sdk-code-examples/local_kumo_engine_graph_widget_preview.py) | Local | Commented preview |
| [Read synthetic CSV and Parquet tables directly and snapshot their graph](cross-industry/sdk-code-examples/local_kumo_engine_local_file_tables.py) | Local | Focused workflow |
| [Retrain a retained CustomerLTV model in a fresh Python process using current source data](cross-industry/sdk-code-examples/local_kumo_engine_retrain_saved_model.py) | Local | Model workflow |
| [Schedule daily local Engine batch predictions on an Airflow worker](cross-industry/sdk-code-examples/local_kumo_engine_airflow_batch_predictions.py) | Local | Focused workflow |
| [Scheduling preview for the MLOps fit/score integration](cross-industry/sdk-code-examples/local_kumo_engine_sagemaker_pipeline.py) | Local | Commented preview |
| [Scheduling preview for the MLOps fit/score integration](cross-industry/sdk-code-examples/snowflake_kumo_engine_scheduled_ml_jobs.py) | Snowflake | Commented preview |
| [Score current source data with the exact trusted bundle produced by the fit job](cross-industry/sdk-code-examples/local_kumo_engine_mlops_score.py) | Local | Model workflow |
| [Turn prediction explanation prompts into natural-language explanations](cross-industry/sdk-code-examples/local_kumo_engine_prediction_prompt_explanations.py) | Local | Focused workflow |

## Energy

### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Regress hourly transformer oil temperature from the public ETT series](energy/sdk-code-examples/local_kumo_engine_transformer_temperature.py) | Local | Model workflow |

## Financial services

### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict reported transaction fraud on Databricks](fsi/notebooks/databricks_kumo_engine_fraud_detection.ipynb) | Databricks | Model workflow |
| [Screen accounts for money muling](fsi/notebooks/databricks_kumo_engine_money_muling_detection.ipynb) | Databricks | Model workflow |
| [Screen synthetic transactions for review](fsi/notebooks/local_kumo_engine_synthetic_transaction_screening.ipynb) | Local | Synthetic binary classification |
| [Detect money mule activity in synthetic accounts](fsi/notebooks/local_kumo_engine_synthetic_money_muling_detection.ipynb) | Local | Synthetic account classification with a baseline and new-account scoring |

### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Classify loan status from the eight-table public Financial relational dataset](fsi/sdk-code-examples/local_kumo_engine_financial_loan_risk.py) | Local | Model workflow |
| [Predict Home Credit default risk from seven public competition tables](fsi/sdk-code-examples/local_kumo_engine_home_credit_default.py) | Local | Model workflow |

## Healthcare

### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict appointment attendance in the next 30 days](healthcare/notebooks/local_kumo_engine_synthetic_appointment_attendance.ipynb) | Local | Synthetic temporal binary classification |

## Hospitality

### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Classify the booking destination of Airbnb new users using sessions and demographics](hospitality/sdk-code-examples/local_kumo_engine_airbnb_booking_destination.py) | Local | Model workflow |
| [Forecast three daily interaction counts for each user](hospitality/sdk-code-examples/local_kumo_engine_restaurant_forecast.py) | Local | Model workflow |
| [Predict each user's interaction count for the next day](hospitality/sdk-code-examples/local_kumo_engine_restaurant_temporal.py) | Local | Model workflow |
| [Predict whether each user will interact in the next day](hospitality/sdk-code-examples/local_kumo_engine_restaurant_binary.py) | Local | Model workflow |
| [Predict which interaction types each user will have in the next day](hospitality/sdk-code-examples/local_kumo_engine_restaurant_multilabel.py) | Local | Model workflow |
| [Rank restaurant relationships without a future prediction window](hospitality/sdk-code-examples/local_kumo_engine_restaurant_static_link.py) | Local | Model workflow |
| [Rank restaurants each user may interact with in the next day](hospitality/sdk-code-examples/local_kumo_engine_restaurant_link.py) | Local | Model workflow |
| [Recommend available recipes from authorized recipe/order tables](hospitality/sdk-code-examples/local_kumo_engine_recipe_recommendations.py) | Local | Model workflow |
| [Reload a trusted restaurant model and score its unchanged original snapshot](hospitality/sdk-code-examples/local_kumo_engine_restaurant_reload.py) | Local | Model workflow |
| [Train a small restaurant model and score twenty held-out cuisine labels](hospitality/sdk-code-examples/local_kumo_engine_restaurant_predict.py) | Local | Model workflow |

## Media and entertainment

### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict low ratings for movies on Snowflake](media/notebooks/snowflake_kumo_engine_netflix_recommendations.ipynb) | Snowflake | Model workflow |
| [Predict the content formats each subscriber will use](media/notebooks/local_kumo_engine_synthetic_content_format_preferences.ipynb) | Local | Synthetic multilabel classification |

### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Predict a movie's average rating during the next twelve months](media/sdk-code-examples/local_kumo_engine_movielens_average_ratings.py) | Local | Model workflow |
| [Recommend Stack posts a user will comment on over the next two years](media/sdk-code-examples/local_kumo_engine_stack_comment_recommendations.py) | Local | Model workflow |
| [Regress MovieLens 1M ratings for user/movie pairs at their rating times](media/sdk-code-examples/local_kumo_engine_movielens_pair_ratings.py) | Local | Model workflow |

## Retail

### Notebooks

| Example | Deployment | Scope |
| --- | --- | --- |
| [Classify store products into a category](retail/notebooks/local_kumo_engine_synthetic_product_category.ipynb) | Local | Synthetic multiclass classification |
| [Explore Kumo Engine tasks with TPC-H on Snowflake](retail/notebooks/snowflake_kumo_engine_tpch_task_examples.ipynb) | Snowflake | Model workflow |
| [Kumo Engine task examples with AdventureWorks in Google Colab](retail/notebooks/colab_kumo_engine_adventureworks_task_examples.ipynb) | Colab | Model workflow |
| [Kumo Engine task examples with TPC-H on Databricks](retail/notebooks/databricks_kumo_engine_tpch_task_examples.ipynb) | Databricks | Model workflow |
| [Predict repeat orders with AdventureWorks](retail/notebooks/snowflake_kumo_engine_adventureworks_repeat_orders.ipynb) | Snowflake | Model workflow |
| [Predict repeat orders with AdventureWorks on Databricks](retail/notebooks/databricks_kumo_engine_adventureworks_repeat_orders.ipynb) | Databricks | Model workflow |
| [Recommend products from synthetic purchase histories](retail/notebooks/local_kumo_engine_synthetic_product_recommendations.ipynb) | Local | Synthetic temporal recommendations |
| [Tag store products with every applicable label](retail/notebooks/local_kumo_engine_synthetic_product_tags.ipynb) | Local | Synthetic multilabel classification |

### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Combine two-part spend predictions with explicit row alignment](retail/sdk-code-examples/local_kumo_engine_paired_queries_evaluate.py) | Local | Focused workflow |
| [Fit H&M lifetime value or recommendation models with supplied example weights](retail/sdk-code-examples/local_kumo_engine_weighted_training.py) | Local | Model workflow |
| [Forecast 28 daily sales values from the public M5 competition](retail/sdk-code-examples/local_kumo_engine_m5_sales_forecasting.py) | Local | Model workflow |
| [Forecast three 30-day order counts for each customer](retail/sdk-code-examples/local_kumo_engine_adventureworks_forecast.py) | Local | Model workflow |
| [Match Walmart and Amazon products using the public labeled entity-matching pairs](retail/sdk-code-examples/local_kumo_engine_product_entity_matching.py) | Local | Model workflow |
| [Predict 28-day customer churn by sales channel in Snowflake](retail/sdk-code-examples/snowflake_kumo_engine_channel_customer_churn.py) | Snowflake | Model workflow |
| [Predict H&M churn, conditional lifetime value, recommendations or daily revenue](retail/sdk-code-examples/local_kumo_engine_hm_customer_predictions.py) | Local | Model workflow |
| [Predict H&M repeat purchase over 90 days and report complementary churn risk](retail/sdk-code-examples/local_kumo_engine_customer_churn.py) | Local | Model workflow |
| [Predict conditional 30-day maximum purchase quantity using the public CustomerLTV mini dataset](retail/sdk-code-examples/local_kumo_engine_customer_lifetime_value.py) | Local | Model workflow |
| [Predict each active customer's order count for the next 90 days](retail/sdk-code-examples/local_kumo_engine_adventureworks_temporal.py) | Local | Model workflow |
| [Predict the size of the next Instacart basket within 30 days](retail/sdk-code-examples/local_kumo_engine_instacart_basket_size.py) | Local | Model workflow |
| [Predict whether an active customer orders again in the next 90 days](retail/sdk-code-examples/local_kumo_engine_adventureworks_binary.py) | Local | Model workflow |
| [Predict whether each active customer orders online, offline, or both](retail/sdk-code-examples/local_kumo_engine_adventureworks_multilabel.py) | Local | Model workflow |
| [Preprocess customer-location entities with read-only Snowflake SQL](retail/sdk-code-examples/snowflake_kumo_engine_sql_preprocessing.py) | Snowflake | Focused workflow |
| [Rank product relationships without a future prediction window](retail/sdk-code-examples/local_kumo_engine_adventureworks_static_link.py) | Local | Model workflow |
| [Rank products each customer may purchase in the next 90 days](retail/sdk-code-examples/local_kumo_engine_adventureworks_link.py) | Local | Model workflow |
| [Recommend coupons from the public Coupon Purchase Prediction competition](retail/sdk-code-examples/local_kumo_engine_coupon_recommendations.py) | Local | Model workflow |
| [Recommend future customers for each product in the public Olist dataset](retail/sdk-code-examples/local_kumo_engine_olist_customer_recommendations.py) | Local | Model workflow |
| [Recommend stores for each user/location combination after explicit preprocessing](retail/sdk-code-examples/local_kumo_engine_custom_recommendation_analysis.py) | Local | Model workflow |
| [Recommend twelve related H&M articles from a prepared article-pair table](retail/sdk-code-examples/local_kumo_engine_hm_item_recommendations.py) | Local | Model workflow |
| [Reload a trusted AdventureWorks model and score a compatible snapshot](retail/sdk-code-examples/local_kumo_engine_adventureworks_reload.py) | Local | Model workflow |
| [Run paired H&M model scripts as local processes and retry only failed fits](retail/sdk-code-examples/local_kumo_engine_paired_queries_poll.py) | Local | Focused workflow |
| [Train an AdventureWorks repeat-order model and score active customers](retail/sdk-code-examples/local_kumo_engine_adventureworks_predict.py) | Local | Model workflow |
| [Train one half of a two-part H&M expected-spend model](retail/sdk-code-examples/local_kumo_engine_paired_queries_train.py) | Local | Model workflow |

## Science

### SDK code examples

| Example | Deployment | Scope |
| --- | --- | --- |
| [Forecast monthly sunspot counts from the public SILSO series](science/sdk-code-examples/local_kumo_engine_sunspot_forecasting.py) | Local | Model workflow |

Commented previews preserve demonstrations of features unavailable in the supported package API.
Their commented demonstrations are not runnable.
Example-specific companion directories hold shared data preparation, helpers and assets; use them through their owning examples.
