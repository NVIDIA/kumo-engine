# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window


def gt_event_metrics(predictions: DataFrame, gt_events: DataFrame, K: int):

    # get predictions
    entity_window = Window.partitionBy('ENTITY').orderBy(F.col("SCORE").desc())
    scores = predictions.withColumn("PREDICTED_RANK",
                                    F.row_number().over(entity_window))

    # get user SET - set of all users with predictions available
    entities = scores.select('ENTITY').dropDuplicates()

    print(f"Total events available in eval window: {gt_events.count()}")

    # cross section with the users we actually made predictions for
    gt_events = gt_events.join(entities.select('ENTITY'), on='ENTITY',
                               how='inner')

    # Get the minimum and maximum order dates
    min_date = gt_events.agg(F.min("TIME_COL")).collect()[0][0]
    max_date = gt_events.agg(F.max("TIME_COL")).collect()[0][0]

    # Print the results
    print("Minimum event date:", min_date)
    print("Maximum event date:", max_date)

    # number of events we are evaluating on
    num_events = gt_events.count()
    print("Number of events we will evaluate on: ", num_events)

    # get active users
    active_entities = gt_events.select('ENTITY').dropDuplicates()

    # only keep scores for active users
    scores = scores.join(active_entities, on='ENTITY', how='inner')

    # make a rank table for GT data
    gt_df = gt_events.select('ENTITY', 'TARGET').distinct().withColumn(
        'LABEL', F.lit(1))
    gt_df.printSchema()

    joined_ranks = scores.join(gt_df, ['ENTITY', 'TARGET'],
                               "left").orderBy('ENTITY', 'SCORE')

    # define the ranking window
    w = Window.partitionBy("ENTITY").orderBy("PREDICTED_RANK").rangeBetween(
        Window.unboundedPreceding, 0)

    # use ranking window to get PR and LABEL_CUMSUM
    joined_ranks = joined_ranks.na.fill({
        'LABEL': 0
    }).withColumn('RECI_RANK',
                  F.col('LABEL') / F.col('PREDICTED_RANK')).withColumn(
                      'LABEL_CUMSUM',
                      F.sum('LABEL').over(w)).withColumn(
                          'PR',
                          F.col('LABEL_CUMSUM') / F.col('PREDICTED_RANK'))

    joined_ranks.printSchema()

    recall_full = joined_ranks.select(
        ['ENTITY', 'LABEL']).groupBy('ENTITY').agg(
            F.sum(F.col('LABEL')).alias('RECALL_FULL')).drop('LABEL')
    joined_ranks_weighted = joined_ranks.filter(
        F.col('PREDICTED_RANK') <= K).withColumn(
            'PR_weight', joined_ranks['PR'] * joined_ranks['LABEL'])

    eval_df_entity = joined_ranks_weighted.groupBy('ENTITY').agg(
        F.sum(F.col('LABEL')).alias('RECALL'),
        F.max(F.col('RECI_RANK')).alias('RR'),
        F.avg(F.col('PR')).alias('AP'),
        F.sum(F.col('PR_weight')).alias('PR_S'))
    # recall_full has each user_id and the recall out of all predictions
    # before filtering to K (RECALL_FULL)
    eval_df_entity = eval_df_entity.join(
        recall_full, on='ENTITY', how='inner').withColumn(
            'WEIGHTED_AVG',
            F.col('PR_S') / (F.least(F.col('RECALL_FULL'), F.lit(K)))).na.fill(
                {'WEIGHTED_AVG': 0})
    eval_df_entity.show()

    # evaluations average over entities
    final_stats = eval_df_entity.agg(
        F.sum(F.col('RECALL')).alias('Recall'),
        F.avg(F.col('RR')).alias('MRR'),
        F.avg(F.col('AP')).alias('MAP'),
        F.avg(F.col('WEIGHTED_AVG')).alias('MAP_weighted')).withColumn(
            'ALL_EVENTS', F.lit(num_events))
    final_stats.show()

    return final_stats
