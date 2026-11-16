# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType


# Define a UDF to compute the cosine similarity between two arrays
def __cosine_similarity(arr1, arr2):
    dot_product = np.dot(arr1, arr2)
    magnitude = np.linalg.norm(arr1) * np.linalg.norm(arr2)
    return float(dot_product / magnitude)


@F.udf(FloatType())
def cosine_similarity(arr1, arr2):
    return __cosine_similarity(arr1, arr2)


# Dot product UDF
@F.udf(FloatType())
def dot_product(arr1, arr2):
    return float(np.dot(arr1, arr2))


# Cosine distance
@F.udf(FloatType())
def cosine_distance(arr1, arr2):
    return 1.0 - __cosine_similarity(arr1, arr2)


# Define a UDF to calculate the norm of the vectors
@F.udf(FloatType())
def l2_distance(arr1, arr2):
    return float(np.linalg.norm(np.subtract(arr1, arr2)))


@F.udf(FloatType())
def l2_norm(x):
    return float(np.linalg.norm(x))


def cosine_comparison(df1_embeds: DataFrame,
                      df2_embeds: DataFrame) -> DataFrame:
    e1_df = df1_embeds.select('ID', F.col('EMBEDDING').alias('EMBEDDING_1'))
    e2_df = df2_embeds.select('ID', F.col('EMBEDDING').alias('EMBEDDING_2'))
    joined_df = e1_df.join(e2_df, 'ID')
    joined_df.show()
    result_df = joined_df.withColumn(
        'cosine', cosine_similarity('EMBEDDING_1', 'EMBEDDING_2'))
    result_df.show()
    return result_df


def l2_comparison(df1_embeds: DataFrame, df2_embeds: DataFrame) -> DataFrame:
    e1_df = df1_embeds.select('ID', F.col('EMBEDDING').alias('EMBEDDING_1'))
    e2_df = df2_embeds.select('ID', F.col('EMBEDDING').alias('EMBEDDING_2'))
    joined_df = e1_df.join(e2_df, 'ID')
    joined_df.show()
    result_df = joined_df.withColumn('l2',
                                     l2_distance('EMBEDDING_1', 'EMBEDDING_2'))
    return result_df
