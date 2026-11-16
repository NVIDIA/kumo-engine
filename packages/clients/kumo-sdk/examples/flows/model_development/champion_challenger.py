# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Support automated retraining pipeline:
1) use the same graph config/model hyperparameters but updated source data
  updated splits
2) compare the champion/challenge on the same holdout
3) make a decision based on some eval
4) register or deploy the winner to the production
"""

import kumoai_studio as kumo
from kumoai_studio.artifact_export import OutputConfig

# Define the connector, table, and graph.
connector = kumo.SnowflakeConnector(
    name="a",
    identity="identity_snowflake",
    warehouse="c",
    database="d",
    schema_name="e",
)

customers = kumo.Table(
    source_table=connector.table("customers"),
    primary_key="customer_id",
)
customers.infer_metadata()

articles = kumo.Table(
    source_table=connector.table("articles"),
    primary_key="article_id",
)
articles.infer_metadata()

transactions = kumo.Table(source_table=connector.table("transactions"))
transactions.infer_metadata()

graph = kumo.Graph({
    'customers': customers,
    'articles': articles,
    'transactions': transactions,
})
graph.link('customers', 'customer_id', 'transactions')
graph.link('articles', 'article_id', 'transactions')

# 1) use the same graph config/model hyperparameters but updated source data
# / updated splits
pquery = kumo.PredictiveQuery(
    graph=graph,
    query=("""PREDICT COUNT(transactions.*, 0, 90, days) > 0
           FOR EACH customers.customer_id
           WHERE COUNT(transactions.price, -60, 0, days) > 2
    """),
)

champion_trainer_id = "production trainer id"
champion_trainer = kumo.Trainer.load(champion_trainer_id)

training_table = pquery.generate_training_table()
model_plan = pquery.suggest_model_plan()
challenge_trainer = kumo.Trainer(model_plan)
challenge_trainer.fit(graph, training_table)

# 2) compare the champion/challenge on the same holdout
# Rerun test for champion_trainer on latest data.
# TODO: Trainer.test() API not yet implemented
# print(f"{champion_trainer.id()} testing on holdout "
#       f"{champion_trainer.test(training_table, graph).metrics}")
# print(f"{challenge_trainer.id()} testing on holdout "
#       f"{challenge_trainer.test(training_table, graph).metrics}")

# 3) make a decision based on some eval
# Customer can make a decision based on metircs.

# 4) register or deploy the winner to the production
# Assume challenge one is better.

prediction_table = pquery.generate_prediction_table()
challenge_trainer.predict(
    prediction_table=prediction_table,
    output_config=OutputConfig(output_types={'predictions'}),
    graph=graph,
)
