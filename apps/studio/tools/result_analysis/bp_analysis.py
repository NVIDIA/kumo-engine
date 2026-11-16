# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from datetime import date, datetime, timedelta

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from .embedding_analysis import cosine_comparison, l2_comparison
from .gt_metrics import gt_event_metrics
from .notebook_output import plot_cosine_similarities
from .prediction_analysis import jaccard_similarity

EMBEDDING_SUB_PATH = 'prediction_output/embeddings.parquet'
PREDICTION_SUB_PATH = 'prediction_output/predictions.parquet'
TARGET = 'TARGET'
ENTITY = 'ENTITY'
TIME_COL = 'TIME_COL'


class Analysis:
    def __init__(self, out_path: str, notebook_output: bool = False):
        builder = SparkSession.builder.config(  # pyright: ignore
            'spark.sql.sources.commitProtocolClass',
            'org.apache.spark.internal.io.cloud.PathOutputCommitProtocol')
        self.spark = builder.appName("test_step").getOrCreate()
        self.try_and_set_path(out_path)
        self.notebook_output = notebook_output

    # check if folder exists and if so append a timestamp to avoid overwrite
    # (spark writes have overwrite mode = False)
    def try_and_set_path(self, out_path):
        sc = self.spark.sparkContext
        fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
            sc._jvm.java.net.URI.create(out_path),
            sc._jsc.hadoopConfiguration(),
        )
        if fs.exists(sc._jvm.org.apache.hadoop.fs.Path(out_path)):
            out_path = out_path + f"_{str(datetime.now())}"
            print("Specified outpath exists, appending current datetime "
                  f"to distinguish: {out_path}")

        self.out_path = out_path

    def load_embeddings(self, path, type):
        embs = self.spark.read.parquet(path)
        if 'COMMON_ID' in type:
            # remove COMMON_ID which will be replace with random connector id
            type_suffix = type.split('.')[1]
            # only filter by part that isn't randomized
            return embs.filter(F.col('TYPE').contains(type_suffix))

        return embs.filter(F.col('TYPE') == type)

    def embedding_dist(self, job1: str, type1: str, job2: str, type2: str,
                       dist_type: str):

        df1_embeds = self.load_embeddings(job1, type1)
        df2_embeds = self.load_embeddings(job2, type2)

        if dist_type == 'cosine':
            cosine_sims = cosine_comparison(df1_embeds, df2_embeds)
            buckets = cosine_sims.select(
                F.floor(F.col('cosine') / 0.01).alias('bucket')).groupBy(
                    'bucket').count().orderBy('bucket')
            quantiles = [0.05, 0.9, 0.95, 0.99]
            quantile_values = cosine_sims.approxQuantile(
                'cosine', quantiles, 0.0001)

            print("Quantiles of Cosine Similarity: "
                  f"{quantiles} {quantile_values}")
            print("Cosine Similarity Buckets:")
            buckets.show()
            if self.out_path:
                buckets.write.parquet(
                    osp.join(self.out_path, 'cosine_similarity_buckets'))
                quantile_df = self.spark.createDataFrame(
                    [[quantiles[i], quantile_values[i]]
                     for i in range(len(quantiles))], ['Quantile', 'Value'])
                quantile_df.write.csv(
                    osp.join(self.out_path, 'cosine_similarity_quantiles'))
            if self.notebook_output:
                plot_cosine_similarities(cosine_sims)
            return quantile_values

        if dist_type == 'l2':
            l2_distances = l2_comparison(df1_embeds, df2_embeds)
            print("L2 Distance Sum:")
            l2_sum = l2_distances.agg(F.sum('l2').alias('l2_sum'))
            l2_sum.show()
            if self.out_path:
                l2_distances.write.parquet(osp.join(self.out_path, 'l2_dists'))
                l2_sum.write.csv(osp.join(self.out_path, 'l2_dists_sum'))
            return l2_sum.collect()[0]['l2_sum']
        return []

    def check_matching_entities(self, job1: str, type1: str, job2: str,
                                type2: str, column_name: str):
        df1_embeds = self.load_embeddings(job1, type1)
        df2_embeds = self.load_embeddings(job2, type2)

        df1_count = df1_embeds.count()
        df2_count = df2_embeds.count()

        df1_distinct_count = df1_embeds.select(column_name).distinct().count()
        df2_distinct_count = df2_embeds.select(column_name).distinct().count()

        df1_minus_df2 = df1_embeds.select(column_name).subtract(
            df2_embeds.select(column_name))
        df2_minus_df1 = df2_embeds.select(column_name).subtract(
            df1_embeds.select(column_name))

        print(f"Total rows in DataFrame1: {df1_count}")
        print(f"Total rows in DataFrame2: {df2_count}")
        print(f"Distinct rows in DataFrame1 for column '{column_name}':"
              f" {df1_distinct_count}")
        print(f"Distinct rows in DataFrame2 for column '{column_name}':"
              f" {df2_distinct_count}")

        if df1_count != df2_count:
            print("Row count does not match between dataframes. Difference:"
                  f" {abs(df1_count - df2_count)}")

        if df1_distinct_count != df2_distinct_count:
            print(f"Distinct row count for column '{column_name}' does not "
                  " match between dataframes. Difference: "
                  f"{abs(df1_distinct_count - df2_distinct_count)}")

        if df1_minus_df2.count() > 0:
            print("Entities in DataFrame1 not in DataFrame2 for column "
                  f"'{column_name}':")
            df1_minus_df2.show()

        if df2_minus_df1.count() > 0:
            print("Entities in DataFrame2 not in DataFrame1 for column "
                  f"'{column_name}':")
            df2_minus_df1.show()

        if self.out_path:
            df1_minus_df2.write.parquet(
                osp.join(self.out_path, 'df1_minus_df2'))
            df2_minus_df1.write.parquet(
                osp.join(self.out_path, 'df2_minus_df1'))

    def prediction_comp(self, job1: str, entity1: str, target1: str, job2: str,
                        entity2: str, target2: str, comp_type: str) -> float:
        df1 = self.spark.read.parquet(job1).withColumnRenamed(
            entity1, ENTITY).withColumnRenamed(target1, TARGET)
        df2 = self.spark.read.parquet(job2).withColumnRenamed(
            entity2, ENTITY).withColumnRenamed(target2, TARGET)

        if comp_type == 'jaccard':
            jaccard_index = jaccard_similarity(df1, df2)
            jaccard_avg = jaccard_index.agg(F.avg(F.col('jaccard_index')))
            print("Avg jaccard index:")
            jaccard_avg.show()
            jaccard_avg = jaccard_avg.withColumnRenamed(
                'avg(jaccard_index)', 'average jaccard index')
            if self.out_path:
                jaccard_index.write.parquet(
                    osp.join(self.out_path, 'jaccard_index'))
                jaccard_avg.write.csv(
                    osp.join(self.out_path, 'jaccard_index_avg'))
            jaccard_avg_score = jaccard_avg.first()
            assert jaccard_avg_score is not None
            return jaccard_avg_score['average jaccard index']
        return -1

    def metrics_event_freq(self, job: str, entity: str, target: str,
                           gt_path: str, gt_target: str, days_eval: int,
                           eval_start: date, event_time_col: str, K: int = 10):
        # load batch predictions
        predictions = self.spark.read.parquet(
            osp.join(job, PREDICTION_SUB_PATH)).drop('PREDICTED') \
            .withColumnRenamed(entity, ENTITY) \
            .withColumnRenamed(target, TARGET)

        embeddings = self.spark.read.parquet(osp.join(job, EMBEDDING_SUB_PATH))
        bp_date = embeddings.first()['TIMESTAMP'].date()
        print("Batch prediction TIMESTAMP: ", bp_date)

        eval_finish = eval_start + timedelta(days=days_eval)
        assert bp_date < eval_start, (
            f"BP date {bp_date} must be earlier than "
            f"evaluation start {eval_start}")

        # prepare ground truth data filtering for selected eval time range
        gt_events = self.spark.read.parquet(gt_path) \
            .select(entity, gt_target, event_time_col) \
            .withColumnRenamed(entity, ENTITY) \
            .withColumnRenamed(gt_target, TARGET) \
            .withColumnRenamed(event_time_col, TIME_COL)

        gt_events = gt_events.filter((F.col(TIME_COL) < eval_finish)
                                     & (F.col(TIME_COL) >= eval_start))

        metrics = gt_event_metrics(predictions, gt_events, K)
        if self.out_path:
            metrics.write.parquet(osp.join(self.out_path, 'gt_metrics'))
