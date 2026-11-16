# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RelBench datasets: wrap relbench tasks for paper benchmarks.

Binary classification tasks:
    relamazon_itemchurn
    relamazon_reviewverified
    relamazon_userchurn
    relarxiv_papercitation
    relavito_searchinfoisuserloggedon
    relavito_searchstreamclick
    relavito_userclicks
    relavito_uservisits
    relevent_event_interestinterested
    relevent_event_interestnot_interested
    relevent_userignore
    relevent_userrepeat
    relf1_driverdnf
    relf1_drivertop3
    relhm_userchurn
    relmimic_patienticulengthofstay
    relratebeer_beerchurn
    relratebeer_brewerdormant
    relratebeer_userchurn
    relstack_postlinkslinktypeid
    relstack_userbadge
    relstack_userengagement
    reltrial_eligibilitiesadult
    reltrial_eligibilitieschild
    reltrial_studieshas_dmc
    reltrial_studyoutcome

Multiclass classification tasks:
    relarxiv_authorcategory
    relsalt_itemincoterms
    relsalt_itemplant
    relsalt_itemshippoint
    relsalt_salesgroup
    relsalt_salesincoterms
    relsalt_salesoffice
    relsalt_salespayterms
    relsalt_salesshipcond
    relstack_badgesclass

Regression tasks:
    relamazon_itemltv
    relamazon_reviewrating
    relamazon_userltv
    relarxiv_authorpublication
    relavito_adctr
    relevent_userattendance
    relevent_usersbirthyear
    relf1_constructor_resultspoints
    relf1_constructor_standingsposition
    relf1_driverposition
    relf1_qualifyingposition
    relf1_resultsposition
    relhm_itemsales
    relhm_transactionsprice
    relratebeer_userbeerrating
    relratebeer_usercount
    relstack_postvotes
    reltrial_sitesuccess
    reltrial_studiesenrollment
    reltrial_studyadverse

Link prediction tasks:
    relamazon_useritempurchase
    relamazon_useritemrate
    relamazon_useritemreview
    relarxiv_paperpapercocitation
    relavito_useradvisit
    relf1_drivercircuitcompete
    relhm_useritempurchase
    relratebeer_userbeerfavorite
    relratebeer_userbeerliked
    relratebeer_userplaceliked
    relstack_postpostrelated
    relstack_userpostcomment
    reltrial_conditionsponsorrun
    reltrial_sitesponsorrun
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from kumoapi.task import TaskType as KumoTaskType
from relbench.base import TaskType as RelTaskType
from relbench.tasks import get_task

import kumoai_studio as kumo

from .benchmarking_dataset import TARGET, BenchmarkingDataset, SupervisedTables
from .relbench_pqueries import RELBENCH_PQUERIES

# Columns excluded when building graphs from the raw RelBench tables.
EXCLUDE = {
    "hm": {
        "article": [  # Duplicated columns:
            "product_type_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
            "product_code",
        ],
    },
    "amazon": {  # Useless text columns:
        "customer": ["customer_name"],
        "product": ["brand", "title", "description"],
        "review": ["review_text", "summary"],
    },
}


@dataclass
class _RelBenchSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _normalize_target_to_binary(target: pd.Series) -> pd.Series:
    """Convert target column to binary 0/1 values."""
    if target.dtype == "object":
        mapping = {"t": 1, "f": 0, True: 1, False: 0}
        target = target.map(mapping).fillna(target)
    elif pd.api.types.is_integer_dtype(target.dtype):
        unique_vals = sorted(target.dropna().unique())
        if len(unique_vals) == 2 and set(unique_vals) != {0, 1}:
            val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            target = target.map(val_map)
    return target


def _encode_targets(
    task,
    y_ctx: pd.Series,
    y_pred: pd.Series | None = None,
    *,
    binary_to_bool: bool,
) -> tuple[pd.Series, pd.Series | None, KumoTaskType]:
    """Encode targets and return encoded series plus Kumo task type."""
    task_type = getattr(task, "task_type", None)

    if task_type == RelTaskType.REGRESSION:
        y_ctx = y_ctx.astype("float64")
        if y_pred is not None:
            y_pred = y_pred.astype("float64")
        kumo_task_type = KumoTaskType.REGRESSION
    elif task_type == RelTaskType.MULTICLASS_CLASSIFICATION:
        all_labels = y_ctx if y_pred is None else pd.concat(
            [y_ctx, y_pred],
            axis=0,
            ignore_index=True,
        )
        classes = sorted(all_labels.dropna().unique())
        class_map = {label: idx for idx, label in enumerate(classes)}
        y_ctx = y_ctx.map(class_map).astype("int64")
        if y_pred is not None:
            y_pred = y_pred.map(class_map).astype("int64")
        kumo_task_type = KumoTaskType.MULTICLASS_CLASSIFICATION
    elif task_type == RelTaskType.BINARY_CLASSIFICATION:
        y_ctx = _normalize_target_to_binary(y_ctx)
        if y_pred is not None:
            y_pred = _normalize_target_to_binary(y_pred)
        if binary_to_bool:
            y_ctx = y_ctx.astype(bool)
            if y_pred is not None:
                y_pred = y_pred.astype(bool)
        else:
            y_ctx = y_ctx.astype("int64")
            if y_pred is not None:
                y_pred = y_pred.astype("int64")
        kumo_task_type = KumoTaskType.BINARY_CLASSIFICATION
    else:
        msg = (f"Unsupported RelBench task_type for "
               f"{getattr(task, 'dataset', None)!r} "
               f"/ {getattr(task, 'name', None)!r}: {task_type!r}")
        raise TypeError(msg)

    return y_ctx, y_pred, kumo_task_type


class RelBenchBenchmarkingDataset(BenchmarkingDataset):
    """Base class for RelBench single-task benchmarks."""

    DATASET_NAME: str = ""
    TASK_NAME: str = ""

    def __init__(self, data_root: str | Path | None = None) -> None:
        # ``data_root`` is accepted for API compatibility
        del data_root

    # -- RelBench helpers ---------------------------------------------

    def _get_relbench_task(self):
        """Return the RelBench task object for this dataset/task pair."""
        try:
            task = get_task(self.DATASET_NAME, self.TASK_NAME, download=True)
        except ValueError:
            # Some autocomplete tasks are constructed on the fly.
            task = get_task(self.DATASET_NAME, self.TASK_NAME)
        return task

    def _get_splits(self) -> _RelBenchSplits:
        """Load train / val / test splits for this task."""
        task = self._get_relbench_task()
        train = task.get_table("train", mask_input_cols=False).df.copy()
        val = task.get_table("val", mask_input_cols=False).df.copy()
        test = task.get_table("test", mask_input_cols=False).df.copy()
        return _RelBenchSplits(train=train, val=val, test=test)

    # -- Single-table benchmarks --------------------------------------

    def get_entity_tables(self) -> SupervisedTables:
        splits = self._get_splits()
        task = self._get_relbench_task()
        target_col = task.target_col
        entity_col = task.entity_col

        # Load the full entity table from the RelBench DB.
        db = task.dataset.get_db(upto_test_timestamp=False)
        entity_table = db.table_dict[task.entity_table]
        entity_df = entity_table.df.copy()
        # The task's entity_col (e.g. "user") may differ from the DB
        # table's pkey_col (e.g. "user_id"). Rename so merges work.
        pkey_col = entity_table.pkey_col
        if pkey_col is not None and pkey_col != entity_col:
            entity_df = entity_df.rename(columns={pkey_col: entity_col})

        # Encode labels jointly for train/val/test so that class mappings
        # (for multiclass) are consistent across splits.
        y_ctx_raw = pd.concat(
            [splits.train[target_col], splits.val[target_col]],
            axis=0,
            ignore_index=True,
        )
        y_test_raw = splits.test[target_col].reset_index(drop=True)
        y_ctx_enc, y_test_enc, kumo_task_type = _encode_targets(
            task,
            y_ctx=y_ctx_raw,
            y_pred=y_test_raw,
            binary_to_bool=False,
        )

        # Build per-event frames with encoded labels.
        df_train_val = pd.concat(
            [splits.train, splits.val],
            axis=0,
            ignore_index=True,
        ).copy()
        df_train_val[TARGET] = y_ctx_enc
        df_test_events = splits.test.copy()
        df_test_events[TARGET] = y_test_enc

        # For training, keep each temporal appearance.
        # The entity table provides static features per entity id, while
        # the target can vary across timestamps.
        time_col = task.time_col
        per_entity_train = df_train_val.dropna(subset=[entity_col])
        if time_col in per_entity_train.columns:
            # Avoid collapsing distinct appearances of the same entity.
            per_entity_train = per_entity_train.drop_duplicates(
                subset=[entity_col, time_col],
                keep="last",
            )
        train_label_cols = [entity_col, TARGET]
        if time_col in per_entity_train.columns:
            train_label_cols.append(time_col)
        train_labels = per_entity_train[train_label_cols].copy()

        # Drop columns that might leak the label (original target column
        # plus any columns listed in task.remove_columns for any table)
        # from the entity-level features.
        entity_df = entity_df.copy()
        if target_col in entity_df.columns:
            entity_df = entity_df.drop(columns=[target_col])
        remove_cols = getattr(task, "remove_columns", []) or []
        leak_cols = {col_name for _tab, col_name in remove_cols}
        drop_cols = [c for c in leak_cols if c in entity_df.columns]
        if drop_cols:
            entity_df = entity_df.drop(columns=drop_cols)

        # Join encoded labels to entity features. The entity table is
        # static, so features are repeated for each temporal label row.
        train_df = train_labels.merge(
            entity_df,
            left_on=entity_col,
            right_on=entity_col,
            how="inner",
        )
        test_label_cols = [entity_col, TARGET]
        if time_col in df_test_events.columns:
            test_label_cols.append(time_col)
        test_df = df_test_events[test_label_cols].merge(
            entity_df,
            left_on=entity_col,
            right_on=entity_col,
            how="inner",
        )

        return SupervisedTables(
            train=train_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
            target_col=TARGET,
            task_type=kumo_task_type,
        )

    def get_joined_tables(self) -> SupervisedTables:
        raise NotImplementedError("Joined tables not implemented")

    # -- Enterprise benchmark -----------------------------------------

    def get_enterprise_graph(self):
        """Build a Kumo Graph mirroring the full RelBench DB schema."""
        dataset = self.DATASET_NAME.lower().removeprefix("rel-")

        # The task accessor downloads and caches the RelBench DB; iterate
        # its native tables and relational metadata directly.
        task = self._get_relbench_task()
        db = task.dataset.get_db(upto_test_timestamp=False)

        tables: dict[str, kumo.Table] = {}
        edges: list[dict[str, str]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            connector = kumo.FileUploadConnector(file_type="parquet")

            for name, table in db.table_dict.items():
                exclude = EXCLUDE.get(dataset, {}).get(name, [])
                df = table.df.drop(
                    columns=[c for c in exclude if c in table.df.columns])

                local_path = tmp_path / f"{name}.parquet"
                df.to_parquet(local_path, index=False)

                if not connector.has_table(name):
                    connector.upload(name=name, path=str(local_path))

                src = connector.table(name)
                tbl = kumo.Table.from_source_table(
                    src,
                    primary_key=table.pkey_col,
                    time_column=table.time_col,
                ).infer_metadata()
                tables[name] = tbl

                for fkey, dst_table in (table.fkey_col_to_pkey_table
                                        or {}).items():
                    edges.append({
                        "src_table": name,
                        "fkey": fkey,
                        "dst_table": dst_table,
                    })

        return kumo.Graph(tables=tables, edges=edges)

    def get_pquery(self) -> str:
        """Predictive query for this RelBench task.

        When a task-specific query is available in RELBENCH_PQUERIES, it
        is used; otherwise we fall back to a generic per-entity
        prediction on the task's target column inferred from the
        RelBench task metadata.
        """
        task = self._get_relbench_task()

        custom = RELBENCH_PQUERIES.get((self.DATASET_NAME, self.TASK_NAME), )
        if isinstance(custom, str):
            return custom

        table = task.entity_table
        target_col = task.target_col
        entity_col = task.entity_col

        return (f"PREDICT {table}.{target_col}\n"
                f"FOR EACH {table}.{entity_col}")


def _make_relbench_class(
    dataset_name: str,
    task_name: str,
) -> type[RelBenchBenchmarkingDataset]:
    """Create a concrete RelBenchBenchmarkingDataset subclass."""
    class_name = (f"RelBench_{dataset_name.replace('-', '')}_"
                  f"{task_name.replace('-', '')}_BenchmarkingDataset")
    attrs: dict[str, object] = {
        "DATASET_NAME":
        dataset_name,
        "TASK_NAME":
        task_name,
        "__doc__": (f"RelBench benchmarking dataset for {dataset_name!r} / "
                    f"{task_name!r}."),
    }
    return type(class_name, (RelBenchBenchmarkingDataset, ), attrs)


def get_all_relbench_datasets() -> dict[str, type[BenchmarkingDataset]]:
    """Return a mapping of dataset name -> RelBench dataset class.

    Keys are suitable for ``--dataset`` in the benchmark scripts and
    follow the pattern:

        {dataset}_{task}

    where ``dataset`` and ``task`` are the raw RelBench names with all
    dashes removed, for example::

        relratebeer_beerrating
    """
    from relbench.tasks import get_task_names, task_registry

    out: dict[str, type[BenchmarkingDataset]] = {}
    for dataset_name, tasks in task_registry.items():
        if not dataset_name.startswith("rel-"):
            continue
        for task_name in get_task_names(dataset_name):
            key = (f"{dataset_name.replace('-', '')}_"
                   f"{task_name.replace('-', '')}")
            out[key] = _make_relbench_class(dataset_name, task_name)
    return out
