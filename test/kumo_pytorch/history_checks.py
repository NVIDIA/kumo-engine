# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Required multi-component histories on verified AML sources and anchors.

These are additional correctness probes, not new model workload definitions.
Declared preflight refusals pass separately from successful execution.
"""

import argparse
import json
import traceback
from collections import Counter
from decimal import Decimal
from pathlib import Path

import kumo.pytorch.api as r

from .aml_inputs import load_selections, verify_inputs
from .aml_source import aml_data_schema
from .anchor_tables import write_probe_anchors
from .involvement import source_reference
from .run_aml import positive, prepare


def history_contract(*, filtered, schema=None):
    events = {}
    for source, role in enumerate(("sender", "receiver")):
        filters = (
            (r.RowFilter("amount_paid", "greater_than", Decimal(0)),)
            if role == "sender"
            else (r.RowFilter("payment_format", "equal", "ACH"),)
        )
        events[role] = r.EventSpec(
            r.Rows(
                (role,),
                r.TimeWindow("event_time", None, -1)
                if role == "sender"
                else r.TimeWindow("event_time", -86401, -3600),
                filters=filters if filtered else (),
            ),
            {
                "transaction": r.Field("transaction_id"),
                "source": r.Constant(source, r.DType("int64")),
            },
        )
    fields = r.Fields(tuple(r.Encode(n, r.Raw()) for n in ("transaction", "source")))
    return r.DataLoaderSpec(
        schema or aml_data_schema(),
        r.SampleSpec(r.EntitySpec("accounts"), events=events),
        r.BatchSpec(
            {
                f"{keep}_{order}": r.Input(
                    r.EventSource(
                        tuple(events),
                        max_events=None if keep == "all" else 3,
                        keep="first" if keep == "all" else keep,
                        order=order,
                    ),
                    fields,
                    r.Packed(),
                )
                for keep in ("all", "first", "last")
                for order in ("ascending", "descending")
            }
        ),
    )


def expected_history(anchor, events, *, filtered, timestamps=False):
    """Independent raw-source membership; no Kumo or source-mapper selection helpers."""
    cutoff = anchor["as_of_time"] // 10**9
    account = anchor["entity_id"]
    selected = []
    for event in events:
        if (
            event["sender"] == account
            and event["time"] < cutoff
            and (not filtered or Decimal(event["values"]["amount_paid"]) > 0)
        ):
            selected.append((event["time"], 0, event["id"]))
        if (
            event["receiver"] == account
            and cutoff - 86400 <= event["time"] <= cutoff - 3600
            and (not filtered or event["values"]["payment_format"] == "ACH")
        ):
            selected.append((event["time"], 1, event["id"]))
    return (
        sorted(selected)
        if timestamps
        else [(source, identity) for _, source, identity in sorted(selected)]
    )


def verify_history_selection(actual, eligible, *, keep, order):
    """Ties may permute or straddle a limit; earlier/later timestamps may not."""
    expected = (
        eligible
        if keep == "all"
        else (eligible[:3] if keep == "first" else eligible[-3:])
    )
    times = {(source, identity): stamp for stamp, source, identity in eligible}
    available = Counter((source, identity) for _, source, identity in eligible)
    assert not (Counter(actual) - available), "Unexpected or duplicate history event"
    observed_times = [times[event] for event in actual]
    required_times = sorted(
        (stamp for stamp, _, _ in expected), reverse=order == "descending"
    )
    assert observed_times == required_times, "History selection/time ordering differs"


def check_histories(
    dataset,
    identities,
    groups,
    reference,
    *,
    directory,
    filtered,
    threads=1,
    schema=None,
):
    directory.mkdir(parents=True, exist_ok=False)
    requests = {
        split: tuple(
            r.SampleRequest(
                row["sample_id"],
                identities[row["entity_id"]],
                row["as_of_time"] // 10**9,
                split=split,
            )
            for row in rows
        )
        for split, rows in groups.items()
    }
    table = write_probe_anchors(
        tuple(q for rows in requests.values() for q in rows),
        directory / "anchors.parquet",
    )
    anchors = {row["sample_id"]: row for rows in groups.values() for row in rows}
    contract = history_contract(filtered=filtered, schema=schema)
    totals = dict.fromkeys(contract.batch.inputs, 0)
    samples = 0
    from .refusals import executing_case

    plan = r.prepare(
        contract,
        dataset,
        r.DiskGraphEngine(index=directory / "index", num_threads=threads),
    )
    with (
        plan,
        executing_case(),
        plan.materialize(
            table,
            fit_split="train",
            encoder_cutoff=max(q.as_of_time for q in requests["train"]) - 1,
        ) as task,
    ):
        for split, split_requests in requests.items():
            seen = []
            for batch in task.batches(split=split, batch_size=4, output_format="model"):
                sample_ids = batch.metadata["sample_ids"]
                assert not batch.metadata["is_padding"].any(), (
                    "Unexpected sample padding"
                )
                for i, sample_id in enumerate(sample_ids):
                    expected = expected_history(
                        anchors[sample_id],
                        reference["events"],
                        filtered=filtered,
                        timestamps=True,
                    )
                    for name, output in batch.inputs.items():
                        keep, order = name.split("_")
                        start, stop = output["offsets"][slice(i, i + 2)].tolist()
                        actual = list(
                            zip(
                                output["source"]["values"][start:stop].tolist(),
                                output["transaction"]["values"][start:stop].tolist(),
                                strict=True,
                            )
                        )
                        verify_history_selection(
                            actual, expected, keep=keep, order=order
                        )
                        totals[name] += len(actual)
                    seen.append(sample_id)
                samples += len(sample_ids)
            assert seen == [q.sample_id for q in split_requests], (
                f"{split}: sample coverage differs"
            )
    assert totals["all_ascending"] > 0, "Real-data probe emitted no events"
    return {"status": "passed", "samples": samples, "events_by_input": totals}


def run_binding(dataset, binding, projection, mapping, directory, *, threads):
    """Reuse source-history probes with a binding's projection and source mapping."""
    from .data import schema_for
    from .population_anchors import projected_rows
    from .refusals import case_failure

    groups = {
        split: [
            dict(
                sample_id=row["sample_id"],
                entity_id=row["registered_entity_id"],
                as_of_time=row["prediction_time_ns"],
            )
            for row in projected_rows(projection, split)
        ]
        for split in ("train", "valid", "test")
    }
    from .aml_source import account_identity_map

    identities = account_identity_map(Path(mapping["table_paths"]["accounts"]).parent)
    reference = source_reference(
        mapping["sources"]["transactions"]["path"],
        [row["entity_id"] for rows in groups.values() for row in rows],
        include_values=True,
        variant=mapping["variant"],
        include_labels=False,
    )
    cases = []
    for filtered in (False, True):
        name = "filters-and-windows" if filtered else "windows"
        try:
            # Each probe has its own preflight; execution errors remain failures.
            result = check_histories(
                dataset,
                identities,
                groups,
                reference,
                directory=directory / name,
                filtered=filtered,
                threads=threads,
                schema=schema_for(dataset, binding),
            )
        except Exception as error:
            result = case_failure(error)
            if result.get("outcome") == "refused":
                result["stage"] = "prepare"
        cases.append({"name": name, **result})
    return {
        "status": "passed" if all(c["status"] == "passed" for c in cases) else "failed",
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors-per-split", type=positive, default=4)
    parser.add_argument("--threads", type=positive, default=1)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "scope": "bounded AML multi-component history correctness",
        "anchors_per_split": args.anchors_per_split,
        "cases": [],
    }
    try:
        files = verify_inputs(args.data_dir)
        selections = load_selections(args.data_dir, limit=args.anchors_per_split)
        dataset, identities, _, _ = prepare(
            args.data_dir, args.output / "prepared", selections, files
        )
        groups = selections["v1"]["forecast_split"]
        reference = source_reference(
            args.data_dir / "datasets/aml-data/v8/hi-small/transactions.parquet",
            [row["entity_id"] for rows in groups.values() for row in rows],
            include_values=True,
        )
        report["inputs"] = list(files.values())
        report["anchors"] = groups
        for filtered in (False, True):
            name = "filters-and-windows" if filtered else "windows"
            try:
                result = check_histories(
                    dataset,
                    identities,
                    groups,
                    reference,
                    directory=args.output / name,
                    filtered=filtered,
                    threads=args.threads,
                )
            except Exception as error:  # noqa: BLE001 - retain every case outcome
                from .refusals import case_failure

                result = case_failure(error)
            report["cases"].append({"name": name, **result})
        report["status"] = (
            "passed"
            if all(c["status"] == "passed" for c in report["cases"])
            else "failed"
        )
    except Exception as error:  # noqa: BLE001 - preparation failures must produce a failing report
        report.update(error=str(error), traceback=traceback.format_exc())
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "### AML multi-source history checks",
        "",
        f"Status: **{report['status']}**",
        "",
    ]
    for case in report["cases"]:
        lines.append(f"- {case['name']}: **{case['status']}**")
        if case.get("outcome") == "refused":
            lines.append(
                "  Passed by explicit preflight refusal; capability was not executed."
            )
            lines.extend(
                f"  {i['code']} at {i['path']}: {i['reason']}"
                for i in case["refusal"]["issues"]
            )
        elif "error" in case:
            lines.extend(["", "```text", case["error"], "```", ""])
        else:
            lines.append(
                f"  Samples: {case['samples']}; emitted events: {case['events_by_input']}"
            )
    if "error" in report:
        lines.extend(["```text", report["error"], "```"])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{report['status']}: {args.output / 'report.md'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
