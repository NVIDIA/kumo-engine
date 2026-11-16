# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Dict, Optional, Union

from kumoapi.pquery import QueryType, ValidatedPredictiveQuery
from kumoapi.typing import ProblemType
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator

from io_utils import kumo_dataclass
from io_utils.json import from_json_dict
from io_utils.yaml import dict_to_yaml
from kumo.config import GraphConfig
from kumo.config.formatting import FieldRequirement, FormatterMetadata
from kumo.config.task import SplitConfigSerializable
from kumo.validation import ValidationResponse


@kumo_dataclass
class PredictiveQueryFieldMetadata(FormatterMetadata):
    """Serves as the Metadata class for kumo's YAML Formatter for the
    Predictive Query YAML.

    condition (FieldRequirement): Whether this field allows and/or requires
        a condition (eg A != B)
    aggregation (FieldRequirement): Whether this field allows and/or requires
        an aggregation (eg AGG(A, B, C))
    require_primary_key (bool = False): Whether to require that the column
        referenced by this field be a primary key. This is useful for entity
        and entity-related fields
    require_primary_key (bool = False): Whether to require that the field
        reference a column (allows fields like train_start_offset: <scalar>)
    """
    condition: FieldRequirement
    aggregation: FieldRequirement
    require_primary_key: bool = False
    require_column: bool = True
    # TODO(frg100): Add a way to make sure entity_filter is in same table as
    # entity


@kumo_dataclass
class PredictiveQueryV2Config:
    """Config class for PredictiveQueryV2.
    This class can be used to create a PredictiveQueryV2 object.
    """
    name: str
    graph: GraphConfig
    validated_predictive_query: ValidatedPredictiveQuery
    timeframe_step: Optional[int] = None
    train_start_offset: Optional[int] = None
    train_end_offset: Optional[Any] = None
    start_time: Optional[int | str] = None
    end_time: Optional[int | str] = None
    split: SplitConfigSerializable = None
    drop_earliest_timeframe: Optional[bool] = True
    forecast_length: Optional[int] = None
    lag_timesteps: Optional[int] = None
    list_distinct_target_is_array: Optional[bool] = False
    lp_bp_layout: Optional[str] = None
    year_over_year: bool = False
    weight_col: Optional[str] = None

    def __post_init__(self):
        # Validation response gets loaded as api.ValidationResponse which lacks
        # functionalities compared to kumo.validation.ValidationResponse
        # So we overwrite the loading here
        if isinstance(self.validated_predictive_query, dict):
            self.validated_predictive_query[
                'validation_response'] = from_json_dict(
                    self.validated_predictive_query['validation_response'],
                    ValidationResponse)
        else:
            if not isinstance(
                    self.validated_predictive_query.validation_response,
                    ValidationResponse):
                self.validated_predictive_query.validation_response = ValidationResponse(
                ).merge(self.validated_predictive_query.validation_response)

    def get_bp_entity_col_name(self) -> str:
        """
        Gets the name of the entity col found in the batch prediction table
        """
        # isort: off
        from kumo.pquery.batch_pred_table_generator import (
            PredictionTableGeneratorV2, )
        # isort: on
        return PredictionTableGeneratorV2.ENTITY

    def query_string(self) -> str:
        """
        Returns the query string for the predictive query.
        """
        return self.validated_predictive_query.to_string()

    @property
    def entity(self) -> str:
        """
        Returns the entity for the predictive query.
        """
        return self.validated_predictive_query.entity_ast.to_string()

    @property
    def target(self) -> str:
        """
        Returns the target for the predictive query.
        """
        return self.validated_predictive_query.target_ast.to_string()

    def options_string(self) -> str:
        """
        Returns the options string for the predictive query.
        """
        arg_dict = {
            'timeframe_step': self.timeframe_step,
            'train_start_offset': self.train_start_offset,
            'train_end_offset': self.train_end_offset,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'split': self.split,
            'forecast_length': self.forecast_length,
            'lag_timesteps': self.lag_timesteps,
            'list_distinct_target_is_array': self.list_distinct_target_is_array,
            'lp_bp_layout': self.lp_bp_layout,
            'year_over_year': self.year_over_year,
            'weight_col': self.weight_col,
        }
        return dict_to_yaml(arg_dict)

    def with_credentials(
            self, credentials: Dict[str, Dict]) -> 'PredictiveQueryV2Config':
        """
        Sets the credentials in the graph and updates the config.
        """
        from kumo.graph import Graph
        from kumo.pquery.predictive_query_validator import _set_pq_credentials
        graph = Graph.from_config(self.graph)
        _set_pq_credentials(
            self.validated_predictive_query,
            graph,
            credentials,
        )
        self.graph = graph.config()
        return self

    @property
    def query_type(self) -> QueryType:
        """
        Returns the query type for the predictive query.
        """
        return self.validated_predictive_query.query_type

    @property
    def is_forecasting(self) -> bool:
        """A backward-compatible property that checks both problem type
        (new approach) and num_forecasts (old approach)"""
        return (
            self.validated_predictive_query.problem_type == ProblemType.FORECAST
            or self.forecast_length is not None and self.forecast_length > 1)


@kumo_dataclass
class PredictiveQueryConfigBase:
    # Metadata
    graph: GraphConfig
    name: str

    # Fields
    # If you add a new field you must update get_formatter_field_metadata and
    # yaml_with_overrides

    # Entity associated with the predictive query.
    entity: Optional[str] = None
    target: Optional[str] = None
    entity_filter: Optional[str] = None  # Serialized Filter object
    temporal_entity_filter: Optional[str] = None  # Serialized Aggregation
    target_filter: Optional[str] = None  # Serialized Filter object
    whatif: Optional[str] = None  # Serialized Aggregation

    # Other options
    # If you add a field update options_to_yaml and default to None to avoid
    # cluttering yaml strs in the db with default values - adding a new arg
    # with a non-none default may break MLOps dashboards in the future.
    forecast_length: Optional[int] = None
    forecast_step: Optional[int] = None  # Deprecated.
    timeframe_step: Optional[int] = None
    train_start_offset: Optional[int] = None
    condition: Optional[str] = None  # Serialized Filter object
    entity_candidate: Optional[Union[bool, str]] = None
    entity_candidate_aggregation: Optional[str] = None
    is_sliding_window: Optional[bool] = False
    lag_timesteps: Optional[int] = None
    lag_window_size: Optional[int] = None
    year_over_year: bool = False

    # Should be of type SplitConfig (needs to be SplitConfigSerializable so
    # it's deserialized as a Dict to make Split.from_config() work.
    # Otherwise it's deserialized as an empty SplitConfig() object and
    # Split.from_config() fails)
    split: Optional[SplitConfigSerializable] = None

    # Arguments for predictive query v2 ###
    # If True, then this is a v2 predictive query
    # So we need to use the `PredictiveQueryV2` object model
    # TODO(jinu): Add a pquery str parameter
    # for pquery v2. Don't split into entity, target
    # and whatif in the config.
    is_pquery_v2: Optional[bool] = False
    top_k: Optional[int] = None
    train_end_offset: Optional[int] = None
    query_string: Optional[str] = None
    query_type: Optional[QueryType] = None

    # Internal flag that tells whether a model has been trained with array
    # targets in link prediction tasks
    list_distinct_target_is_array: Optional[bool] = False
    lp_bp_layout: Optional[str] = None
    start_time: Optional[int | str] = None
    end_time: Optional[int | str] = None

    def to_predictive_query_v2_config(
        self,
        credentials: Optional[Dict[str, Dict]] = None,
    ) -> 'PredictiveQueryV2Config':
        """Converts a PredictiveQueryV2Config object to a
        PredictiveQueryV2Config object.

        Args:
            credentials (Dict[str, Dict]): credentials of all connectors,
                given as a mapping from the connector name to its credentials.
        """
        from kumo.graph import Graph
        from kumo.pquery.utils import validator_kwargs
        assert self.query_string is not None
        parsed_query = PQLParser().to_parsed_predictive_query(self.query_string)
        graph = Graph.from_config(self.graph)
        validator = PredictiveQueryValidator(
            **validator_kwargs(graph, query_string=self.query_string))
        predictive_query, response = validator.validate_predictive_query(
            parsed_query)
        if not response.ok:
            # Should not happen unless the config is corrupted
            raise ValueError(response.message())
        assert predictive_query is not None
        return PredictiveQueryV2Config(
            name=self.name,
            graph=self.graph,
            validated_predictive_query=predictive_query,
            timeframe_step=self.timeframe_step,
            train_start_offset=self.train_start_offset,
            train_end_offset=self.train_end_offset,
            start_time=self.start_time,
            end_time=self.end_time,
            split=self.split,
            forecast_length=self.forecast_length,
            lag_timesteps=self.lag_timesteps,
            year_over_year=self.year_over_year,
            list_distinct_target_is_array=self.list_distinct_target_is_array,
            lp_bp_layout=self.lp_bp_layout,
        )


def preprocess_pq_yaml(yaml_str: str) -> str:
    """Automatically fix common errors in yaml syntax to provide a better
    experience to the user.
    """
    # Replace "key:value" with "key: value", as this is required by the yaml
    # spec, but is a common error from users. We only handle the case when the
    # key is a valid identifier, because it is not safe to perform this
    # correction in the general case when the key contains special characters.
    return re.sub(r'^(\s*[\w_]+:)(?=\S)', r'\1 ', yaml_str, flags=re.MULTILINE)
