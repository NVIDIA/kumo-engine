# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rel-stack dataset with user-post-comment recommendation task.

Requirements:
    * Modify `API_URL` and `API_KEY` to point to a live, appropriate instance
    * Modify `OUTPUT_DIR` and `OPENAI_API_KEY` to run LLM embedding job
"""
import os

import pooch
from relbench.base import Database
from relbench.datasets import get_dataset

import kumoai_studio as kumo
from kumoai_studio.artifact_export import OutputConfig
from kumoai_studio.connector import upload_table
from kumoai_studio.trainer.job import BatchPredictionJobResult

API_URL = "https://demo-sdk.kumoai.cloud/api"
API_KEY = ""  # Insert key

# Initialize the Kumo SDK:
kumo.init(api_key=API_KEY, url=API_URL)

dataset = get_dataset('rel-stack')
db: Database = dataset.get_db()
path = os.path.join(pooch.os_cache("relbench"), 'rel-stack', 'db')

connector = kumo.FileUploadConnector(file_type="parquet")

# Upload tables to Kumo dataplane
if not connector.has_table("badges"):
    upload_table(name="badges", path=os.path.join(path, "badges.parquet"))
if not connector.has_table("comments"):
    upload_table(name="comments", path=os.path.join(path, "comments.parquet"))
if not connector.has_table("postHistory"):
    upload_table(name="postHistory", path=os.path.join(path,
                                                       "postHistory.parquet"))
if not connector.has_table("postLinks"):
    upload_table(name="postLinks", path=os.path.join(path,
                                                     "postLinks.parquet"))
if not connector.has_table("posts"):
    upload_table(name="posts", path=os.path.join(path, "posts.parquet"))
if not connector.has_table("users"):
    upload_table(name="users", path=os.path.join(path, "users.parquet"))
if not connector.has_table("votes"):
    upload_table(name="votes", path=os.path.join(path, "votes.parquet"))

# Create Tables
badges = kumo.Table.from_source_table(connector['badges'], primary_key='Id',
                                      time_column="Date")
badges.infer_metadata()
comments = kumo.Table.from_source_table(connector['comments'],
                                        primary_key='Id',
                                        time_column="CreationDate")
comments.infer_metadata()
postHistory = kumo.Table.from_source_table(connector['postHistory'],
                                           primary_key='Id',
                                           time_column="CreationDate")
postHistory.infer_metadata()
postLinks = kumo.Table.from_source_table(connector['postLinks'],
                                         primary_key='Id',
                                         time_column="CreationDate")
postLinks.infer_metadata()
posts = kumo.Table.from_source_table(connector['posts'], primary_key='Id',
                                     time_column="CreationDate")
posts.infer_metadata()
users = kumo.Table.from_source_table(connector['users'], primary_key='Id',
                                     time_column="CreationDate")
users.infer_metadata()
votes = kumo.Table.from_source_table(connector['votes'], primary_key='Id',
                                     time_column="CreationDate")
votes.infer_metadata()

# Create Graph
graph = kumo.Graph(
    tables={
        'badges': badges,
        'comments': comments,
        'postHistory': postHistory,
        'postLinks': postLinks,
        'posts': posts,
        'users': users,
        'votes': votes
    }, edges={
        kumo.Edge('comments', 'UserId', 'users'),
        kumo.Edge('comments', 'PostId', 'posts'),
        kumo.Edge('badges', 'UserId', 'users'),
        kumo.Edge('postLinks', 'PostId', 'posts'),
        kumo.Edge('postLinks', 'RelatedPostId', 'posts'),
        kumo.Edge('postHistory', 'UserId', 'users'),
        kumo.Edge('postHistory', 'PostId', 'posts'),
        kumo.Edge('votes', 'UserId', 'users'),
        kumo.Edge('votes', 'PostId', 'posts'),
        kumo.Edge('posts', 'OwnerUserId', 'users'),
        kumo.Edge('posts', 'ParentId', 'posts'),
        kumo.Edge('posts', 'AcceptedAnswerId', 'posts'),
    })
graph.validate()

# Predict a list of 10 posts each user will comment on in the next 2 years
# Note that might not be exactly the same task as defined by Relbench
pquery = kumo.PredictiveQuery(
    graph=graph,
    query="""
        PREDICT LIST_DISTINCT(comments.PostId, 0, 730, days) RANK TOP 10
        FOR EACH users.Id
    """,
)
pquery.validate()

train_table = pquery.generate_training_table()

model_plan = pquery.suggest_model_plan()

print(model_plan)

trainer = kumo.Trainer(model_plan)
training_job = trainer.fit(
    graph=graph,
    train_table=train_table,
    non_blocking=False,
)
print(f"Training metrics: {training_job.metrics()}")

pred_table = pquery.generate_prediction_table()
prediction_job = trainer.predict(
    graph=graph,
    prediction_table=pred_table,
    output_config=OutputConfig(output_types={'predictions'}),
    non_blocking=False,
)
assert isinstance(prediction_job, BatchPredictionJobResult)
prediction_job.summary()
