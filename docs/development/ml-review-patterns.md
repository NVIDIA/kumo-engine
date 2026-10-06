# ML Pod Review Patterns

This document captures specific patterns and examples from past ML pod reviews that inform the review guidelines.

> **Living Document**: This file is updated as new PR reviews provide additional insights.

## Pattern 1: Use Existing Helpers

**Reviewer comment:** "Use helper functions from kumo-api ValidatedPredictiveQuery class"

**What triggers it:**
Reimplementing logic that already exists in utility files or helper classes across the codebase. Common examples:
- Custom AST walkers when `ValidatedPredictiveQuery` already has methods
- Reimplementing `table_name()`, `col_name()`, `fqn()` helpers
- Duplicating logic from existing util modules

**Example from review:**
```python
# Custom implementation flagged:
def _infer_target_table_from_condition(self) -> Optional[str]:
    def _table_from_node(node: ASTNode | None) -> Optional[str]:
        # ... recursive AST walking ...

# Should check ValidatedPredictiveQuery methods first
```

**General rule:** Before writing new utility logic, search for existing helpers in the relevant module and in `kumo-api`.

---

## Pattern 2: isinstance vs hasattr

**Reviewer comment:** "Should this massive any instead be a check if isinstance(Table, TrainingTableConfig)?"

**What triggers it:**
```python
# Fragile (flagged):
if any(
    hasattr(table, attr)
    for attr in ('weight_col', 'target_col', 'entity_include_bit_col', 'is_pred')
):

# Robust:
if isinstance(table, TrainingTableConfig):
```

---

## Pattern 3: Leakage Prevention

**Reviewer comment:** "Current aggregation implementation for weights is actually causing leakage because it's aggregating the full history of the data, not the training data."

**What triggers it:**
Aggregating data without respecting time boundaries.

**Key insight:** Leakage is unlikely to be caught in e2e testing but causes silent degradation in production.

**Pattern to follow:**
```python
# Ensure aggregations are bounded by training time window
# Same time window as target aggregation
```

---

## Pattern 4: Empty Aggregation Handling

**Reviewer comment:** "What's the intended behaviour for aggregations that are undefined on an empty interval? E.g. AVG or MAX? This will drop training examples."

**What triggers it:**
Using aggregations like AVG/MAX/MIN/FIRST/LAST without handling empty input.

**Considerations:**
- AVG of no rows = NULL → drops training example
- MAX/MIN of no rows = NULL → drops training example
- FIRST/LAST of no rows = NULL → drops training example
- COUNT of no rows = 0 → keeps training example

**Resolution:** Document intended empty-input behavior and cover it with regression tests; choose aggregations supported by the current PQL validator.

---

## Pattern 5: Feature Flag Behavior

**Reviewer comment:** "This will just cause batch predictions to fail with no good explanations if we train a model with the flag and then turn it off"

**What triggers it:**
```python
# Flagged - silently changes behavior:
if list_distinct_target_is_array and not allow_array_targets():
    logger.warning("forcing False")  # Silent degradation

# Original - clear error:
if list_distinct_target_is_array and not allow_array_targets():
    raise ValueError("Add ALLOW_ARRAY_TARGETS flag...")
```

**Pattern:** Feature flags should maintain backward compatibility, not silently degrade.

---

## Pattern 6: File Organization

**Reviewer comment:** "Let's move validations out of this file. train_table_generator.py is already too massive."

**What triggers it:**
Adding validation logic to already-large files.

**Pattern:**
- `train_table_generator.py` → Move validations to `pquery_validators/`
- New cohesive utilities like `weight_spec.py` are acceptable

---

## Pattern 7: Test Coverage

**Reviewer comment:** "Needs more testing"

**Required coverage dimensions:**
1. All relevant task types (not just the one being added)
2. All relevant aggregation types
3. All relevant filter types (static AND temporal)
4. DiskGraph compiler and service-adapter coverage, plus affected connector paths where applicable

**Reviewer comment:** "Why are existing tests being dropped?"

**Pattern:** Tests should not be removed without justification. If a test is obsolete, explain why in the PR.

---

## Pattern 8: Documentation

**Reviewer comment:** "Missing docstring" / "docstring"

**What triggers it:**
New functions/classes without documentation.

**Expected format:**
```python
def _get_weight_spec(self) -> Optional[WeightSpec]:
    """Parse weight_col into a WeightSpec.

    Returns:
        Parsed WeightSpec if weight_col is set; otherwise None.

    Raises:
        TrainingTableGenerationError: If weight_col format is invalid.
    """
```

---

## Pattern 9: UI Code Generation

**Reviewer comment:** "A lot of these interfaces should already be created by code-gen... you can trigger orval regeneration"

**What triggers it:**
Manually defining types that exist in generated code (`ui/src/gen/`).

**Pattern:** Use `/trigger-ui-codegen` comment or run `make generate-ui-client-code`.

---

## Cross-Reference: Existing Columns

**Reviewer tip:** "If you're trying to weight the link prediction table by frequency, PQ already computes the column into the training table under the name FREQUENCY_COL_NAME"

**Check before adding new columns:**
- `FREQUENCY_COL_NAME` - already computed for link prediction
- `ENTITY_INCLUDE_BIT` - entity filtering

Reference: `kumo/pquery/train_table_generator.py`

---

## Pattern 10: Don't Extend to Unrelated Areas

**Reviewer comment:** "Let's not touch baselines as part of this PR. Baselines for regression won't automatically generalise to baselines for forecasting."

**What triggers it:**
```python
# Flagged - extending to adjacent functionality:
if self.task.task_type in [
        TaskType.REGRESSION,
        TaskType.FORECASTING,  # <-- Added without validation
]:
```

**Pattern:** Just because code paths look similar doesn't mean they'll work the same. Extending functionality requires its own validation and should be a separate PR.

---

## Pattern 11: Technical Restrictions Have Reasons

**Reviewer comment:** "Let's not extend forecasting to these aggregations yet. There is a reason why they're omitted: we cannot impute missing values, i.e. an empty interval with COUNT aggregation is 0, but an empty interval for AVG or MAX is undefined."

**What triggers it:**
```python
# Flagged - removing restriction without understanding why:
if rhs_target.aggr not in [
        AggregationType.SUM, AggregationType.COUNT,
        AggregationType.COUNT_DISTINCT, AggregationType.AVG,  # Added
        AggregationType.MIN, AggregationType.MAX,  # Added
]:
```

**Pattern:** Before extending supported operations, understand why they were restricted. Document the reasoning if you do remove restrictions.

---

## Pattern 12: Follow-up PRs for Uncertain Changes

**Reviewer comment:** "Let's do this one as a follow-up, as I'm not sure that it will work out of the box. And if it does, it's an easy PR."

**What triggers it:**
Changes that might work but haven't been validated.

**Pattern:** If a change is uncertain, make it a separate PR. This:
- Keeps the main PR focused
- Allows independent testing
- Makes it easier to revert if issues arise

---

## Pattern 13: Premature Changes

**Reviewer comment:** "This change will only be needed after the PQL syntax has been updated."

**What triggers it:**
```python
# Flagged - change depends on unreleased work:
if self.task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
    if self.task_type == TaskType.REGRESSION and self.is_sliding_window:
        # This FORECASTING branch won't be hit until PQL syntax changes
```

**Pattern:** Don't add code paths that can't be exercised yet. Wait for dependent changes to land first.

---

## Pattern 14: PR Scope Feedback

**Reviewer comment:** "This PR mostly touches things that don't need touching, and misses items that need to be done."

**What triggers it:**
- Scope creep into unrelated areas
- Missing the core changes that were requested
- Touching too many files without clear connection to goal

**Pattern:** Before submitting, verify:
1. Every change directly supports the stated goal
2. You haven't drifted into "while I'm here" changes
3. The core requirements are actually addressed

---

## Pattern 15: E2E / Integration Testing Required

**Reviewer comment:** "Could you paste an example of an end to end run of forecasting either manual or some e2e test"

**What triggers it:**
New features or significant changes without demonstrated proof. Especially when changes touch multiple layers (e.g. rest + pquery), integration testing is needed since some bugs are only caught across layer boundaries.

**Pattern:** Include in PR description or comments:
- Link to a successful end-to-end validation run
- Screenshot/logs of manual test run
- For multi-step flows, show each stage working

---

# Source PRs

These patterns were extracted from actual ML pod reviews:

| PR | Feature | Primary Reviewers |
|----|---------|-------------------|
| #25572 | weight_col for training tables | vid-koci, wsad1, adi-kumo |
| #26438 | FORECASTING task type support | vid-koci, wsad1 |
