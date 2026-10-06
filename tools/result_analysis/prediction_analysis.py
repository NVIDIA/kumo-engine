# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


# target and entity columns are renamed to constants TARGET and ENTITY
# in Analysis.prediction_comp before calling this method to simplify handling
# here.
def jaccard_similarity(df1: DataFrame, df2: DataFrame) -> DataFrame:
    intersection_df = df1.alias("df1").join(df2.alias("df2"),
                                            (F.col("df1.ENTITY") ==
                                            F.col("df2.ENTITY")) &
                                            (F.col("df1.TARGET") ==
                                            F.col("df2.TARGET")), "inner") \
        .groupBy("df1.ENTITY") \
        .agg(F.count("*").alias("intersection_size"))
    intersection_df.show()
    union_df = df1.select("ENTITY", "TARGET").union(
        df2.select("ENTITY", "TARGET")) \
        .distinct() \
        .withColumn("dummy", F.lit(1)) \
        .groupBy("ENTITY", "dummy") \
        .agg(F.count("*").alias("union_size"))
    result_df = intersection_df.join(union_df, "ENTITY")
    result_df = result_df.withColumn(
        "jaccard_index",
        F.col("intersection_size") / F.col("union_size")).drop("dummy")
    return result_df


def edit_distance(df1: DataFrame, df2: DataFrame) -> DataFrame:
    return df2
