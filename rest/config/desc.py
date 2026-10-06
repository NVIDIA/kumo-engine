# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import markdown

overview = markdown.markdown(
    """The Kumo REST API enables you to integrate training and batch
    prediction into your recurrent production workflows.<br>
    <br>For any training and predictive task, you should always first start
    a training job by writing a Predictive Query using our UI. Each
    training task you start on the UI is associated with an ID, which we
    call the `predictive_query_id` in our REST API.
    After creating the Predictive Query in the UI,
    you can make calls to our public REST API for any
    recurring prediction pipeline. We currently support two use cases:<br>
    <ol>
    <li>You might have new data coming in to your data source on a regular
    basis. If you would like the Predictive Queries to train with the new
    data, simply do so by appending to/replacing the data in the same
    location of your data source, then call `start_training` with the REST
    API. Kumo will automatically ingest the new data into our Table format,
    and make updates to the Graph your Predictive Query is written on.</li>
    <li>If you would like to make recurring batch prediction (e.g. on a
    daily basis), the Kumo REST API supports calls to make
    Batch Predictions. Every call to make a Batch Prediction will first
    re-ingest the data from your data source, then make predictions based
    on the Predictive Query and the current date/time.</li>
    </ol>""")

start_training = markdown.markdown(
    """Launch a job to (re-)train the Predictive Query
    on refreshed Table data.""")

cancel_train = markdown.markdown(
    """Cancel a running training job associated with the given
    `predictive_query_id`. Cancellation is an asynchronous "best effort"
    operation. Users can call `GET`
    "/v1/training_jobs/{training_job_id}" to verify if
    the training job has been successfully cancelled. No action will be
    taken in the following cases:<br>
    <ul>
    <li>No training job is actively running.</li>
    <li>The job has already been completely, regardless of whether the
    training succeeded or failed.</li>
    </ul>""")

delete_train = markdown.markdown(
    """Delete a finished training job associated with the given
    `predictive_query_id`. Deletion is unsafe and irrevesible. If the training
    job successfully produced model artifacts, the model and all metadata will
    be deleted. Non-successful HTTP Status code will be returned in following
    cases:<br>
    <ul>
    <li>404: If the predictive query or training job is not found.</li>
    <li>422: If the training job status is still RUNNING.</li>
    </ul>""")

get_train = markdown.markdown(
    """Get the training job status of a Predictive Query. If the job is
    still running, the response will report detailed progress. If the job
    has completed successfully, the response will contain a summary of
    job execution and evaluation metrics.""")

list_train = markdown.markdown(
    """List the running and finished training jobs of a Predictive Query in
    reverse chronological order. Use `job_status` parameter to filter jobs by
    status. Use `limit` parameter to set max number of jobs returned (or all
    jobs if not specified). `limit` must be a positive integer (`>= 1`);
    non-positive values are rejected with a 422.""")

start_pred = markdown.markdown(
    """Use the `predictive_query_id` to start a Batch Prediction job. All
    Tables will be re-ingested from your data source, and the associated
    Graphs will be updated, so that the job produces predictions over
    the most current dataset.""")

cancel_pred = markdown.markdown(
    """Cancel a running batch prediction job associated with the
    `predictive_query_id`. Cancellation is an asynchronous "best effort"
    operation. Users can call `GET`
    "predictive_queries/{predictive_query_id}/prediction_status" to
    verify if the job has been successfully cancelled. No action will be
    taken in the following cases:<br>
    <ul>
    <li>No Batch Prediction job is actively running.</li>
    <li>The Batch Prediction job has already been completely, regardless
    of whether the prediction succeeded or failed.</li>
    </ul>""")

pred_status = markdown.markdown(
    """Get the current Batch Prediction job status of a Predictive Query.
    If the job is still running, the response will report detailed progress.
    If the job has been completed successfully,
    the response will contain a summary of the job execution
    and output data paths.""")

progress = markdown.markdown(
    """Get the current Batch Prediction job progress of a Predictive Query.""")

list_pred = markdown.markdown(
    """List the running and finished batch prediction jobs of a Predictive
    Query in reverse chronological order. Use `job_status` parameter to filter
    jobs by status. Use `limit` parameter to set max number of jobs returned
    (or all jobs if not specified). `limit` must be a positive integer
    (`>= 1`); non-positive values are rejected with a 422.""")

list_all_jobs = markdown.markdown(
    """List all running and finished jobs in reverse chronological order.
    Use `query` parameter to filter jobs by status, type, or time range.""")

delete_pred = markdown.markdown(
    """Delete a finished batch prediction job associated with the given
    `predictive_query_id`. Deletion is unsafe and irrevesible. If the batch
    prediction job was successful, output result stored within Kumo will be
    deleted and no longer available for export. Non-successful HTTP Status code
    will be returned in following cases:<br>
    <ul>
    <li>404: If the predictive query or batch prediction job is not found.</li>
    <li>422: If the batch prediction job status is still RUNNING.</li>
    </ul>""")

_force_graph_rename_desc = """
If set to true, this will force setting a graph to a name alias, even if the
graph already has a different name or if there is collision with another
graph's name alias; otherwise 422 error code will be returned to forbid
renaming and protect alias/name collision.  Default to false if not specified.
"""

_force_table_rename_desc = """
If set to true, this will force setting a table to a name alias, even if the
table already has a different name or if there is collision with another
table's name alias; otherwise 422 error code will be returned to forbid
renaming and protect alias/name collision.  Default to false if not specified.
"""

_skip_graph_validation_desc = """
If set to true, this will skip graph validation after creation. Only used for
callers that have already validated the graph before calling graph creation.
"""
