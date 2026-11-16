# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator
from yaml.scanner import ScannerError

from io_utils import logging
from kumo.graph import Graph
from kumo.pquery.utils import validator_kwargs

log = logging.getLogger(__name__)


def validate_query_string(
    query_string: str,
    graph: Optional[Graph] = None,
) -> Optional[str]:
    """
    This function validates the PredictiveQuery based on
    the dialect
    Args:
        query_string (str): PredictiveQuery string.
        graph (Optional[Graph]): Graph object.
    """
    try:
        parsed_query = PQLParser().to_parsed_predictive_query(query_string)
        validator = PredictiveQueryValidator(
            **validator_kwargs(Graph.from_config(graph),
                               query_string=query_string))
        _, response = validator.validate_predictive_query(parsed_query)
        if not response.ok:
            return response.message()
        else:
            return None
    except ValueError as err:
        return str(err)
    except ScannerError:
        return 'Invalid YAML, must only have one field and one value per line'
