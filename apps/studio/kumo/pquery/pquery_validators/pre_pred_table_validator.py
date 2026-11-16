# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from kumo_core.training.validation import validate_anchor_time

from io_utils import logging
from kumo.graph import Graph
from kumo.graph.table import Table
from kumo.pquery.pquery_validators.train_table_validator import MAX_RHS_ROWS
from kumo.validation.validation import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)

logger = logging.getLogger(__name__)


class PrePredTableValidator():
    """Runs validations that need to be run before the batch
    prediction table generation is started.

    """
    def validate_anchor_time(
        self,
        anchor_date: pd.Timestamp,
        min_date: pd.Timestamp | None,
        target_table_name: str,
    ) -> ValidationResponse:
        r"""Validation of the prediction anchor time. Produces error if
        anchor date is before the earliest timestamp in target table.

        Args:
            anchor_date (pd.Timestamp):
                The Prediction anchor time.
            min_date (pd.Timestamp | None]:
                The smallest date in target table. (default: :obj:`None`)
            target_table_name (str):
                The target table name.

        Returns:
            ValidationResponse: Errors encountered.
        """
        response = ValidationResponse()
        error = validate_anchor_time(anchor_date, min_date, target_table_name)
        if error is not None:
            response.errors.append(error)
        return response

    def validate_anchor_time_forecasting(
        self,
        anchor_date: pd.Timestamp,
        timeframe_step: pd.Timedelta,
        lag_timesteps: int,
        target_table_name: str,
        max_date: pd.Timestamp | None,
        min_date: pd.Timestamp | None,
    ) -> ValidationResponse:
        r"""Returns a `ValidationResponse` with errors when one of the two
        thing are true:
        1. If number of timeframes in `anchor_date`-`min_date` is less
        than `lag_timesteps`, then older lag data is not available
        for forecasting.
        2. If number of timeframes in `max_date`-`anchor_date`
        is greater than `timeframe_step`,
        then latest lag data is not available for forecasting.
        Args:
            anchor_date(Timestamp): The anchor date for batch prediction.
            timeframe_step(pd.Timedelta): The timframe step used during
                training.
            lag_timesteps(int): The number of lag features for forecasting.
            target_table_name(str): the target table used to generate train
                table.
            min_date(Timestamp, optional): min date in target table.
            max_date(Timestamp, optional): max_date in target table.
        """
        response = ValidationResponse()
        if min_date is not None:
            assert min_date < anchor_date
            if (anchor_date - min_date) / timeframe_step < lag_timesteps:
                response.errors.append(
                    ValidationError(
                        title="Older lag data is missing",
                        message=f"Prediction anchor time {anchor_date} is too "
                        f"close to the start ({min_date}) of data in "
                        f"{target_table_name}, so earlier lag features will "
                        "be missing."))

        if max_date is not None:
            if anchor_date > max_date and (anchor_date -
                                           max_date) / timeframe_step >= 1:
                response.errors.append(
                    ValidationError(
                        title="Latest lag data is not present",
                        message=f"Prediction anchor time {anchor_date} is past "
                        f"the data end ({max_date}) in {target_table_name}, "
                        "so latest lag features will be empty."))
        return response

    def validate_num_rhs_rows(self, target_table_name: str, graph: Graph,
                              is_child_job: bool):

        validation = ValidationResponse()

        target_table = graph.table(target_table_name)
        if isinstance(target_table, Table):
            rhs_num_nodes = target_table.num_rows()
            if rhs_num_nodes > MAX_RHS_ROWS:
                title = "Target table has too many unique rows"
                message = (
                    f"Target table '{target_table_name}' has "
                    f"{rhs_num_nodes:,} unique rows, which exceeds "
                    f"the {MAX_RHS_ROWS:,} row limit for link prediction "
                    "batch predictions. (This limit only applies "
                    "to predictions, not embeddings.)\n")
                if is_child_job:
                    message += f"To resolve the issue, please reduce the target table ('{target_table_name}') size."
                    validation.errors.append(
                        ValidationError(
                            title=title,
                            message=message,
                        ))
                else:
                    message += "To avoid a failure in the batch prediction job "
                    "when generating predictions, consider reducing the target table size."
                    validation.warnings.append(
                        ValidationWarning(
                            title=title,
                            message=message,
                        ))

        return validation
