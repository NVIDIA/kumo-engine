from typing import List, Optional, Union
from kumoapi.pquery.validated_predictive_query import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Constant,
    Filter,
    Join,
    LogicalOperation,
)
from ._c import DiskGraphTTBuilder, EdgeRef, RowFilter, StageFilter

class DiskGraphPlanCompiler:
    """
    Translates a ValidatedPredictiveQuery (AST) into a DiskGraphTTPlan (JSON).
    """

    def __init__(self, builder: Optional[DiskGraphTTBuilder] = None):
        self.builder = builder or DiskGraphTTBuilder()

    def compile(
        self,
        query: ValidatedPredictiveQuery,
        timestamps: List[int],
        forecast_length: int = 1,
        lag_timesteps: Optional[int] = None,
    ) -> str:
        """
        Build the DiskGraph execution plan.

        Args:
            query: The validated PQL query.
            timestamps: A list of discrete epoch microsecond anchor times.
            forecast_length: Steps forward for temporal validity.
            lag_timesteps: Steps backward for temporal validity.

        Returns:
            A JSON string representing the DiskGraphTTPlan.
        """
        # 1. Temporal Spec
        if query.query_type.name == "TEMPORAL" and timestamps:
            self.builder.temporal(timestamps, forecast_length, lag_timesteps)

        # 2. Entity Spec
        entity_table, entity_col = query.entity_column.split('.')
        self.builder.for_each(entity_table, entity_col)

        # TODO: Handle entity_ast filters (push to self.builder.entity_filter)

        # 3. Prediction Type
        # By default, assume Node prediction. Link prediction logic will be added.
        self.builder.node_prediction()

        # 4. Target AST Traversal
        # We will walk the target_ast to build stages. For now, a placeholder.
        # self._traverse_target(query.target_ast)

        return self.builder.build()

    def _traverse_target(self, ast: ASTNode):
        """Recursively translates target AST nodes into DiskGraph operations."""
        pass
