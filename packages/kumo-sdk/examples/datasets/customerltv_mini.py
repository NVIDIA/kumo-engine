# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CustomerLTV Mini (Toy Dataset) on Kumo SDK.

Requirements:
    * Modify `API_URL` and `API_KEY` to point to a live, appropriate instance
    * Pass `--custom_pred_table_path` with an S3 URI that the
      `KUMO_S3_IDENTITY_ID` identity can write to and read from
"""
import argparse
import os
import uuid

from kumoapi.jobs import MetadataField

import kumoai_studio as kumo
from kumoai_studio.artifact_export import OutputConfig
from kumoai_studio.encoder import TextEmbedding
from kumoai_studio.pquery import PredictionTable
from kumoai_studio.trainer.job import BatchPredictionJobResult

parser = argparse.ArgumentParser()
parser.add_argument('--training_job_id', type=str, default='')
parser.add_argument(
    '--export_loc', type=str, default='s3', choices=['s3', 'db'],
    help='Export location for prediction table'
    'See `export_connector`` in the script to see'
    'exact location.')
parser.add_argument(
    '--custom_pred_table_path', type=str,
    default=('s3://example-bucket/sdk/examples/customerltv/'
             'custom/prediction/table.parquet'),
    help='S3 URI where the custom prediction table is written and read back')
args = parser.parse_args()

API_URL = "http://localhost:10002"
API_KEY = "test:DISABLED"
KUMO_PROJECT = os.getenv("KUMO_PROJECT")
DATABRICKS_IDENTITY_ID = os.getenv("KUMO_DATABRICKS_IDENTITY_ID")
S3_IDENTITY_ID = os.getenv("KUMO_S3_IDENTITY_ID")

# Initialize the Kumo SDK:
kumo.init(url=API_URL, api_key=API_KEY, project=KUMO_PROJECT)
PROJECT_ID = kumo.current_project().id

# Create a connector to a backing data store:
assert S3_IDENTITY_ID
connector = kumo.S3Connector(
    name="customerltv_source",
    identity=S3_IDENTITY_ID,
    root_dir="s3://kumo-public-datasets/customerltv_mini_integ_test/",
)

pred_output_table_name = 'customer_ltv_mini'
if args.export_loc == 's3':
    assert S3_IDENTITY_ID
    export_connector = kumo.S3Connector(
        name="customerltv_output",
        identity=S3_IDENTITY_ID,
        root_dir="s3://kumo-public-datasets/customerltv_mini_integ_test/",
    )
elif args.export_loc == 'db':
    assert DATABRICKS_IDENTITY_ID
    export_connector = kumo.DatabricksConnector(
        name="COMMON_ID1",
        identity=DATABRICKS_IDENTITY_ID,
        warehouse_id="82d859e4a6b283a0",
        catalog="kumo_test_catalogue",
        schema_name="pred_tables",
    )
    pred_output_table_name = ('pred_tables', 'customer_ltv_mini')

# Metadata ####################################################################

customer_table = "customer"
customer_id = "CustomerID"
stock_table = "stock"
stock_code = "StockCode"
transaction_table = "transaction"
invoice_date = "InvoiceDate"
quantity = "Quantity"
unit_price = "UnitPrice"

# Training ####################################################################

job_id = args.training_job_id
if job_id == "":
    customer = kumo.Table.from_source_table(
        source_table=connector.table(customer_table),
        primary_key=customer_id,
    ).infer_metadata()
    customer.print_definition()

    stock = kumo.Table.from_source_table(
        source_table=connector.table(stock_table),
        primary_key=stock_code,
    ).infer_metadata()
    stock.print_definition()

    transaction = kumo.Table.from_source_table(
        source_table=connector.table(transaction_table),
        time_column=invoice_date,
    ).infer_metadata()
    transaction['Description'].stype = kumo.Stype.text
    transaction[invoice_date].dtype = kumo.Dtype.date
    transaction.print_definition()

    # Define graph:
    graph = kumo.Graph(
        tables={
            customer_table: customer,
            stock_table: stock,
            transaction_table: transaction,
        },
        edges=[
            dict(src_table=transaction_table, fkey=stock_code,
                 dst_table=stock_table),
            dict(src_table=transaction_table, fkey=customer_id,
                 dst_table=customer_table),
        ],
    )
    graph.validate()

    # Save graph as template for the future. The graph can then be loaded with
    #   >> graph = kumo.Graph.load('customerltv_graph')
    # graph.save('customerltv_graph')

    # Create predictive query:
    pquery = kumo.PredictiveQuery(
        graph=graph,
        query=(
            f"PREDICT MAX({transaction_table}.{quantity}, 0, 30)\n"
            f"FOR EACH {customer_table}.{customer_id}\n"
            f"ASSUMING SUM({transaction_table}.{unit_price}, 0, 7, days) > 15"
        ),
    )
    print(f"Task type: {pquery.get_task_type()}")

    # Save PQ as template for the future. The PQ can then be loaded with
    #   >> pquery = kumo.PredictiveQuery.load('customerltv_pquery')
    # pquery.save('customerltv_pquery')

    # Snapshot graph, wait for it to finish. Note that this is not a necessary
    # step in the pipeline:
    snapshot_id = graph.snapshot(non_blocking=False, force_refresh=True)

    # Get edge health stats, which should be ready by now since we are blocking
    # on the snapshot
    edge_health = graph.get_edge_stats(non_blocking=False)
    print(f"Edge health: {edge_health}")

    full_stats = graph.get_table_stats(wait_for="full")
    print(f"Full stats: {full_stats}")

    # Generate training table:
    train_table = pquery.generate_training_table(non_blocking=True)

    # Train!
    model_plan = pquery.suggest_model_plan()
    model_plan.training_job.num_experiments = 3
    model_plan.column_processing.encoder_overrides = {
        f'{transaction_table}.Description': TextEmbedding('glove.6B-50d')
    }
    print(model_plan)
    trainer = kumo.Trainer(model_plan)
    training_job = trainer.fit(
        graph=graph,
        train_table=train_table,
        non_blocking=False,
    )
    print(f"Training metrics: {training_job.metrics()}")
    job_id = training_job.job_id

# Inference ###################################################################

print(f"Generating predictions from training job {job_id}")

# Load the training job, PQ, and Graph:
training_job = kumo.TrainingJob(job_id, project_id=PROJECT_ID).result()
trainer = kumo.Trainer.load(job_id, project_id=PROJECT_ID)
pquery = kumo.PredictiveQuery.load_from_training_job(job_id,
                                                     project_id=PROJECT_ID)
graph = pquery.graph

# Try with new S3 connector, so it has a new FPID:
assert S3_IDENTITY_ID
prediction_connector = kumo.S3Connector(
    name='customerltv_prediction_source',
    identity=S3_IDENTITY_ID,
    root_dir='s3://kumo-public-datasets/customerltv_mini_integ_test/temp/',
)
graph['customer'].source_table = prediction_connector['customer']

# Use a new predictive query, which will show a warning but will
# work:
pquery = kumo.PredictiveQuery(
    graph=graph,
    query=(f"PREDICT MAX({transaction_table}.{quantity}, 0, 30)\n"
           f"FOR EACH {customer_table}.{customer_id}\n"),
)

# Predict!
pred_table = pquery.generate_prediction_table(non_blocking=True)
prediction_job = trainer.predict(
    graph=graph,
    prediction_table=pred_table,
    output_config=OutputConfig(
        output_connector=connector,
        output_types={'predictions', 'embeddings'},
        output_table_name='kumo_predictions',
        output_metadata_fields=[MetadataField.JOB_TIMESTAMP],
    ),
    training_job_id=training_job.job_id,
    non_blocking=False,
)
assert isinstance(prediction_job, BatchPredictionJobResult)
print(f'Batch prediction job summary: {prediction_job.summary()}')

# Export predictions to a separate place:
# Keep this value and reuse it if submitting this same export must be retried.
export_request_id = str(uuid.uuid4())
export_job = prediction_job.export(
    output_config=OutputConfig(
        output_types={'predictions'},
        output_connector=export_connector,
        output_table_name=pred_output_table_name,
        output_metadata_fields=[MetadataField.JOB_TIMESTAMP],
    ),
    non_blocking=False,
    request_id=export_request_id,
)
print(f'Export artifact job {export_job}')

# Custom Inference ############################################################

# Write a custom table, then address it through a project-owned S3
# connector. Any-URI mode allows the full path to be supplied directly,
# including arbitrarily deep bucket subpaths.
pred_table_df = pred_table.result().data_df()
custom_pred_table_df = pred_table_df.head(100).astype(
    {'TIMESTAMP': 'datetime64[ms]'})
custom_pred_table_path = args.custom_pred_table_path
custom_pred_table_df.to_parquet(custom_pred_table_path)
custom_source_connector = kumo.S3Connector(
    name='customerltv_custom_prediction_source',
    identity=S3_IDENTITY_ID,
    root_dir=None,
)
custom_pred_table = PredictionTable(
    source_table=custom_source_connector.table(custom_pred_table_path))
custom_prediction_job = trainer.predict(
    graph=graph,
    prediction_table=custom_pred_table,
    output_config=OutputConfig(
        output_types={'predictions', 'embeddings'},
        output_connector=connector,
        output_table_name='kumo_predictions_custom',
    ),
    training_job_id=training_job.job_id,
    non_blocking=False,
)
assert isinstance(custom_prediction_job, BatchPredictionJobResult)
print(f'Batch prediction job summary: {custom_prediction_job.summary()}')
