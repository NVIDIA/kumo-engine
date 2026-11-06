# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict 28-day customer churn by sales channel in Snowflake.
Synthetic data only.
Generate synthetic example data first:
    python synthetic-data/engine_retail/generate.py --dataset fashion --output-dir ./synthetic-fashion
Use that directory as --source-root or --input-dir. Metrics are demonstration only.
Manually load these files into your own Snowflake schema, using uppercase table and column names.
For fashion, load transactions.parquet as TRANSACTIONS; customers/articles keep their names.
The generator writes local files only. Configure your own SPCS environment or account credentials.
Install: pip install "kumo-engine[snowflake,text]"
"""
import json
import os
from pathlib import Path
import pandas as pd
from kumo.engine import (Edge, FileConnector, Graph, ModelOutputConfig,
                         PredictiveQuery, SnowflakeConnector, Table, Trainer, configure)


def main():
    output = Path(os.environ['KUMO_OUTPUT_DIR']).resolve()
    scratch = Path(os.environ['KUMO_SCRATCH_DIR']).resolve()
    if output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError('Scratch and retained output must be separate.')
    output.mkdir(parents=True, exist_ok=False)
    configure(scratch_dir=scratch)
    source = SnowflakeConnector.from_spcs_env()
    customers = Table.from_source_table(source['CUSTOMERS'], primary_key='CUSTOMER_ID')
    articles = Table.from_source_table(source['ARTICLES'], primary_key='ARTICLE_ID')
    entities = Table.from_source_table(source.sql("""
        SELECT TO_JSON(ARRAY_CONSTRUCT(c.CUSTOMER_ID, t.SALES_CHANNEL_ID)) AS PK,
               c.CUSTOMER_ID AS CUSTOMER_ID, t.SALES_CHANNEL_ID AS SALES_CHANNEL_ID
        FROM CUSTOMERS c CROSS JOIN (SELECT DISTINCT SALES_CHANNEL_ID FROM TRANSACTIONS) t
    """), primary_key='PK')
    transactions = Table.from_source_table(source.sql("""
        SELECT CUSTOMER_ID, ARTICLE_ID, SALES_CHANNEL_ID, T_DAT,
               TO_JSON(ARRAY_CONSTRUCT(CUSTOMER_ID, SALES_CHANNEL_ID)) AS FK
        FROM TRANSACTIONS
    """), time_column='T_DAT')
    graph = Graph({'CUSTOMERS':customers, 'ARTICLES':articles,
                   'ENTITIES':entities, 'TRANSACTIONS':transactions},
                  [Edge('TRANSACTIONS','CUSTOMER_ID','CUSTOMERS'),
                   Edge('TRANSACTIONS','ARTICLE_ID','ARTICLES'),
                   Edge('TRANSACTIONS','FK','ENTITIES')])
    graph.validate()
    snapshot = graph.snapshot()
    query = PredictiveQuery(graph, 'PREDICT COUNT(TRANSACTIONS.*, 0, 28, days) = 0 FOR EACH ENTITIES.PK')
    training = query.generate_training_table(snapshot)
    trainer = Trainer(query.suggest_model_plan(snapshot))
    result = trainer.fit(training, snapshot=snapshot,
        export_config=ModelOutputConfig(FileConnector(root_dir=str(output)), path='model'))
    print('Evaluation metrics (synthetic, demonstration only):', result.metrics)

    # Batch predictions
    population = query.generate_prediction_table(snapshot, plan=trainer.training_table_generation_plan)
    expected = pd.read_parquet(population.table_path)['ENTITY1']
    if len(expected) < 2:
        raise ValueError('At least two customer/channel rows are required.')
    scores = trainer.predict(population, snapshot=snapshot, binary_classification_threshold=0.5).frame
    if len(scores) != len(expected) or set(scores['ENTITY']) != set(expected):
        raise ValueError('Scores must match the customer/channel population.')
    # Structured JSON keys preserve both IDs without fixed-length substrings.
    scores['SALES_CHANNEL_ID'] = scores['ENTITY'].map(lambda value: json.loads(value)[1])
    scores = scores.sort_values(['SALES_CHANNEL_ID', 'True_PROB'], ascending=[True, False])
    scores['CHANNEL_RANK'] = scores.groupby('SALES_CHANNEL_ID').cumcount() + 1
    top = scores.loc[scores['CHANNEL_RANK'] <= 1000]
    print('True_PROB is churn probability; rank 1 is highest risk within a channel.')
    print(top.head(10).to_string(index=False))
    scores.to_parquet(output / 'channel_churn_predictions.parquet', index=False)
    top.to_parquet(output / 'top_1000_per_channel.parquet', index=False)
    # Predicted labels are not observed churn, so these ranks do not establish recall.


if __name__ == '__main__':
    main()
