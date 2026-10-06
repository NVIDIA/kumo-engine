# DiskGraph Batch Prediction: Architectural Design & Evolution

This document chronicles the architectural design decisions, semantic challenges, and implementation iterations undertaken to integrate Kumo's Batch Prediction (BP) PQL suite natively into the DiskGraph engine.

## 1. The Core Objective
The primary objective was to entirely bypass Kumo's legacy distributed PySpark execution layer during Batch Prediction. By executing Static Node Predictions, Temporal Node Predictions, Link Prediction Candidate Extractions, and Forecasting natively over memory-mapped Rust indices, we aimed to achieve $O(1)$ memory streaming and dramatic throughput improvements.

## 2. Decoupled Target Null Emitting (Static Node Prediction)
In standard Kumo PySpark orchestration, a Static Node Prediction job only generates predictions for entities that are "unlabeled" (i.e., they have no historical events in the target window).

### The Challenge: Math vs. Structure
*   **PySpark's Approach:** PySpark executes the target AST to generate a `target_df`. Users with no events are dropped by the `INNER JOIN`. PySpark then does a `LEFT JOIN` back to the base entity table, artificially injecting `NULL` for missing users. It then applies an `IS NULL` filter to keep them.
*   **DiskGraph's Challenge:** DiskGraph performs single-traversal execution. If we rely on standard SQL aggregators, an empty window mathematically evaluates to `0` (for `COUNT`) or `0.0` (for `SUM`). If we applied the `IS NULL` filter, it would drop all unlabeled users!
*   **Failed Iteration:** We initially tried forcing the Rust aggregators to return `Value::Null` for empty windows. This correctly triggered the `IS NULL` filter, but catastrophically broke 89 Rust unit tests because intermediate mathematical comparisons (like `WHERE COUNT(...) < 5`) suddenly received `None` instead of `0`, violating strict SQL logic.

### The Solution: `Op::HasEdges`
We completely decoupled **Topological Existence** from **Mathematical Aggregation**:
1.  **Rust Core**: We introduced `Op::HasEdges`, an ultra-fast $O(1)$ short-circuiting operator that simply returns a boolean `true` the moment it finds a matching edge, bypassing math entirely.
2.  **Python Compiler**: When the compiler detects a Static Node Prediction, it doesn't execute the mathematical target. Instead, it extracts the path/window and injects `HasEdges`.
3.  **Filtration**: It then applies a native `StageFilter.compare(has_edges_alias, "==", False)` to strictly identify "unlabeled" users, perfectly mimicking PySpark's outer-join reconstruction without corrupting mathematical invariants.

## 3. Link Prediction Candidate Extraction (The "Ghost Drop" Paradox)
Link Prediction requires extracting a "Candidate Pool" (RHS) to pair against the active users (LHS). In Kumo, the user's AST often defines this via an edge traversal: `LIST_DISTINCT(edge.fk_to_item) WHERE edge.clicked = True`.

### The Challenge: Edge vs. Node Tables
*   **Failed Iteration (Iterating the Edge Table):** We initially programmed the compiler to iterate over the literal leaf of the AST (the Edge Table). The Rust engine extracted the foreign key (`item_id`), but immediately passed it to `is_canonical_node`. Because `item_id` was not the physical Primary Key of the edge table (which is `EVENT_ID`), the engine assumed it was a non-canonical duplicate and dropped **100% of the candidates**.
*   **Failed Iteration (The `is_pk` bypass hack):** We injected an `is_pk=False` parameter into the Rust Engine to disable deduplication. This produced the right number of rows, but outputted millions of duplicates (because it iterated the denormalized edge table), completely skewing the PyG candidate space.
*   **Failed Iteration (Reverse Edges):** We attempted to use `HasEdges` to iterate the Target Node Table and traverse backward to evaluate the `edge.clicked = True` filter. We realized this was overly complex and unnecessary because Kumo already handles filter sanitization.

### The Solution: Context Swapping & `BPFilterTransformer`
We discovered that Kumo's PySpark orchestrator uses a semantic engine called `BPFilterTransformer` to structurally guarantee that **all edge-specific filters are dropped** before Candidate Extraction (because historical edge interactions shouldn't filter a global, future-facing candidate pool).

1.  **Abstractions Restored**: We completely reverted all Rust "bypass" hacks. The Rust engine strictly requires iterating over canonical Primary Keys.
2.  **Target Transposition**: The Python orchestrator (`validate_bp_e2e.py`) now reads the physical graph metadata, discovers the true Target Node Table, and passes it explicitly to the compiler via `lp_rhs_entity_table`.
3.  **Sanitized AST**: The orchestrator uses Kumo's native `drop_joins()` utility to strip edge filters, passing only pure node-level constraints to the compiler.
4.  **Result**: The compiler explicitly swaps the execution context. DiskGraph iterates the Destination Node Table directly, natively evaluating `is_canonical_node` against the true PKs, producing a perfectly unique, correctly filtered candidate pool at lightning speed.

## 4. The PyArrow Streaming HStack
For Legacy Static Link Prediction, Kumo requires combining the LHS and RHS into a single table via horizontal stacking (`zip_longest` in PySpark), where the shorter dataframe (usually RHS candidates) is padded by repeating its first row.

*   **The Challenge**: Using Pandas `concat` on a 370-million row dataset (`akihironitta`) caused immediate Out-Of-Memory (OOM) failures.
*   **The Solution**: We implemented a zero-copy PyArrow streaming architecture. We completely materialize the small RHS candidate pool into an in-memory `pa.array`. As the massive LHS stream yields 100k-row chunks, we dynamically slice, tile, and append the RHS `TARGET` column to the `RecordBatch`, writing directly to `.parquet`. This dropped peak memory from ~35GB to less than ~50MB, processing 370M rows in ~6 minutes.

## 5. Forecasting & Parallel Array Materialization
Forecasting in Kumo requires building a target array spanning multiple historical windows.
*   **The Solution**: We built `Op::Autoregression` and `OutputMode::Forecasting` directly into the Rust execution engine. DiskGraph recursively evaluates the core target AST across an arbitrary array of `lag_offsets` and `yoy_offsets`, emitting a dense, multi-dimensional ListArray (`AUTOREGRESSION_COL`). This was successfully tested on 370M users, evaluating 11.1 Billion localized graph queries sequentially in a single memory space.

## Conclusion
The DiskGraph BP implementation represents a mathematically rigorous translation of PySpark's distributed relational algebra into localized, memory-mapped graph traversal primitives. By maintaining strict architectural boundaries—where the Python orchestrator handles schema discovery and sanitization, and the Rust engine handles pure canonical execution—we achieved 100% data parity with production output at unprecedented scale.

## 6. Python Orchestrator Execution Patterns
This section dictates the exact contract and invocation sequence the Python orchestrator (e.g., `diskgraph_batch_pred.py`) must follow when calling the DiskGraph execution API (`compiler.compile` and `execute_tt_plan`) for each ML Task Type.

### Node Prediction (Binary / Multiclass)
*   **Compilation**: 
    *   Call `compiler.compile(..., task_type=task_type.value, is_prediction=True)`. 
    *   *No special overrides required.* The compiler automatically injects the `HasEdges` structural target filter if `timestamps` are empty (Static).
*   **Execution**:
    *   Iterate over `execute_tt_plan`.
    *   **Post-Processing**: 
        *   Drop `entity_id_target` and `TARGET` from the returned PyArrow batches.
        *   If the query is purely `STATIC`, drop the `TIMESTAMP` column (unless explicitly overridden by `extra_fields`).
        *   Inject a monotonically increasing `arange` integer column across all batches.

### Multilabel Ranking
*   **Compilation**:
    *   Call `compiler.compile(..., task_type=task_type.value, is_prediction=True)`.
    *   Handled structurally identical to standard Node Prediction. If a prediction time is passed (e.g., a fallback snapshot), the engine natively drops nodes created after the snapshot via internal temporal boundaries.
*   **Execution**: 
    *   Same as Node Prediction. Drop `TARGET` column.
    *   If `STATIC`, drop `TIMESTAMP`.
    *   Inject `arange`.

### Link Prediction (Separated vs HStack BP Layout)
Link Prediction BP layout is controlled by `lp_bp_layout`. The legacy
`predictive_query.list_distinct_target_is_array` flag still maps to the old
coupled behavior for compatibility:

* `list_distinct_target_is_array=True` maps to `lp_bp_layout="separate_lhs_rhs"`.
* `list_distinct_target_is_array=False` maps to `lp_bp_layout="hstack"`.

This BP layout choice is independent from LP training target shape. Training
target shape is still controlled by `list_distinct_target_is_array` (true → array
target, false → exploded rows); choosing a BP layout does not force a particular
training shape.

#### 1. Separated LHS/RHS Mode (`lp_bp_layout="separate_lhs_rhs"`)
*   **Compilation**: Call `compiler.compile(..., task_type=task_type.value, is_prediction=True)`.
*   **Execution**: Single-pass execution. Output targets natively. Drop `TIMESTAMP` if static, and inject `arange`.

#### 2. Legacy HStack Mode (`lp_bp_layout="hstack"`)
*   **Dual-Pass Architecture**: The orchestrator must perform two explicit compilation and execution passes.
*   **LHS Compilation (Entities)**:
    *   Call `compiler.compile(query=query, task_type=task_type.value, lp_prediction_mode="LHS", is_prediction=True)`.
    *   The engine yields the `ENTITY` column natively. (If `QueryType == STATIC`, the Rust Engine *automatically* injects `__PSEUDO_TIME__` into the output; the Python orchestrator does not need to drop/re-alias the timestamp here).
*   **RHS Compilation (Candidate Extraction)**:
    *   **Sanitization**: The orchestrator must use Kumo's `BPFilterTransformer` to sanitize edge-level constraints out of the target AST, producing a `transformed_query`.
    *   **Context Swapping**: The orchestrator parses the `predictive_query_config.graph` to find the actual Destination Node Table and its Primary Key.
    *   Call `compiler.compile(query=transformed_query, task_type=task_type.value, lp_prediction_mode="RHS", lp_rhs_entity_table=dest_table, lp_rhs_entity_column=dest_pk)`.
*   **Streaming HStack Execution**:
    *   Materialize the entire RHS iterator into memory as a `pa.array` (it is inherently unique).
    *   Stream the LHS iterator in chunks.
    *   For each LHS chunk, dynamically tile/pad the RHS array (repeating the first RHS candidate) to match the LHS chunk size using `pyarrow.concat_arrays`.
    *   Append `arange` sequentially to the augmented batches and flush to Parquet.

### Forecasting
*   **Compilation**:
    *   The orchestrator uses Kumo's `get_lag_timestamps` and `get_yoy_timestamps` to calculate standard temporal steps based on the user's prediction time.
    *   Convert these timestamps to relative offset seconds.
    *   Call `compiler.compile(..., task_type=task_type.value, lag_offsets_secs=[...], is_prediction=True)`.
*   **Execution**:
    *   The engine automatically outputs parallel `AUTOREGRESSION_COL` arrays.
    *   **Post-Processing**: Drop `TARGET` column. Inject `arange` across the sequence, and stream the multi-dimensional arrays directly to disk.
