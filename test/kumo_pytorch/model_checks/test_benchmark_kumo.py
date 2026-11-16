"""Run the generic adapter on non-AML Kumo data and an unrelated linear model."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from kumo.pytorch import api as r

from test.kumo_pytorch.benchmarking import BatchCounts, run
from test.kumo_pytorch.benchmarking.kumo import KumoInputs, kumo_workload


def product_state_factory(config, directory):
    """Importable independent dataset adapter for subprocess/state experiments."""

    def inputs(recorder):
        schema = r.DataSchema(
            (
                r.TableSpec(
                    "products",
                    "id",
                    {"id": r.DType("int64"), "price": r.DType("float64")},
                    time_column=r.NON_TEMPORAL,
                ),
            )
        )
        contract = r.DataLoaderSpec(
            schema,
            r.SampleSpec(
                r.EntitySpec("products"), entity_values={"price": r.Field("price")}
            ),
            r.BatchSpec(
                {
                    "features": r.Input(
                        r.EntitySource(),
                        r.Fields((r.Encode("price", r.ZScore()),)),
                        r.Separate(),
                    )
                }
            ),
        )
        anchors = tuple(
            r.SampleRequest(f"item-{i}", i, 100, split="train") for i in range(4)
        )
        return KumoInputs(
            contract,
            r.Dataset(
                {
                    "products": r.InMemoryRows(
                        [{"id": i, "price": float(i)} for i in range(4)]
                    )
                }
            ),
            r.InMemoryEngine(),
            anchors,
            anchors,
            materialize_options={"fit_split": "train"},
            batch_options={"batch_size": 3},
        )

    return kumo_workload(
        identity={"dataset": "products", "workload": "price"},
        inputs=inputs,
        count=lambda batch: BatchCounts(len(batch.metadata["sample_ids"]), 0),
        load_fitted_state=config.get("load_fitted_state"),
        save_fitted_state=config.get("save_fitted_state"),
    )


def test_native_state_comparison_in_fresh_processes(tmp_path):
    from test.kumo_pytorch.benchmarking.campaign import run_campaign

    result = run_campaign(
        "test.kumo_pytorch.model_checks.test_benchmark_kumo:product_state_factory",
        {},
        tmp_path / "campaign",
        repetitions=2,
        epochs=2,
        compare_fitting=True,
    )
    assert result["independent_repeated_runs"] == "measured", result
    assert all(row["status"] == "measured" for row in result["fitting_comparisons"]), (
        result
    )
    assert result["status"] == "incomplete"


@pytest.mark.parametrize(
    "backend,training",
    (
        ("in_memory", False),
        ("in_memory", True),
        ("diskgraph", False),
        ("diskgraph", True),
        ("polars", False),
    ),
)
def test_unrelated_dataset_and_model(tmp_path, backend, training):
    models, seen = [], []

    def factory(config, workdir):
        def inputs(recorder):
            schema = r.DataSchema(
                (
                    r.TableSpec(
                        "products",
                        "product_id",
                        {"product_id": r.DType("int64"), "price": r.DType("float64")},
                        time_column=r.NON_TEMPORAL,
                    ),
                )
            )
            contract = r.DataLoaderSpec(
                schema,
                r.SampleSpec(
                    r.EntitySpec("products"), entity_values={"price": r.Field("price")}
                ),
                r.BatchSpec(
                    {
                        "features": r.Input(
                            r.EntitySource(),
                            r.Fields((r.Encode("price", r.Raw()),)),
                            r.Separate(),
                        )
                    }
                ),
            )
            rows = [{"product_id": i, "price": float(i + 1)} for i in range(5)]
            recorder.measure("source.rows", len(rows))
            if backend in {"diskgraph", "polars"}:
                with recorder.span("source.convert"):
                    path = workdir / "products.parquet"
                    pq.write_table(pa.Table.from_pylist(rows), path)
                dataset = r.Dataset({"products": r.Parquet(str(path))})
                if backend == "polars":
                    pytest.importorskip("polars")
                    from test.kumo_pytorch.reference_engine.polars import PolarsEngine

                    engine = PolarsEngine()
                else:
                    engine = r.DiskGraphEngine(index=workdir / "index", num_threads=1)
                recorder.measure("converted.bytes", path.stat().st_size)
            else:
                dataset = r.Dataset({"products": r.InMemoryRows(rows)})
                engine = r.InMemoryEngine()
            anchors = tuple(
                r.SampleRequest(f"product-{i}", i, 100, split="train") for i in range(5)
            )
            return KumoInputs(
                contract,
                dataset,
                engine,
                anchors,
                anchors,
                materialize_options={"fit_split": "train"},
                reference=backend == "polars",
                batch_options={
                    "batch_size": 3,
                    "output_format": "model"
                    if backend in {"diskgraph", "polars"}
                    else "inspectable",
                },
            )

        def count(batch):
            ids = batch.metadata["sample_ids"]
            seen.extend(ids)
            # This small fixture checks the actual adapter boundary, not family dispatch.
            assert batch.inputs["features"]["price"]["values"].tolist() == [
                float(int(s.rsplit("-", 1)[1]) + 1) for s in ids
            ]
            tensors = list(batch.inputs["features"]["price"].values())
            return BatchCounts(
                len(ids),
                int(batch.metadata["is_padding"].sum()),
                output_elements=sum(t.numel() for t in tensors),
                output_bytes=sum(t.numel() * t.element_size() for t in tensors),
            )

        def consumer(task, recorder):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(0.25)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            models.append(model)

            def step(batch, recorder):
                values = (
                    batch.inputs["features"]["price"]["values"].float().reshape(-1, 1)
                )
                optimizer.zero_grad()
                with recorder.span("model.forward"):
                    prediction = model(values)
                with recorder.span("model.loss"):
                    loss = (prediction - 2 * values).square().mean()
                with recorder.span("model.backward"):
                    loss.backward()
                with recorder.span("model.optimizer"):
                    optimizer.step()
                return True

            return step

        return kumo_workload(
            identity={
                "dataset": "product-fixture",
                "workload": "price",
                "backend": backend,
                "model": "linear" if training else None,
            },
            inputs=inputs,
            count=count,
            consumer_factory=consumer if training else None,
        )

    result = run(factory, {}, tmp_path / "run", repetitions=1, epochs=2)
    assert result["execution_status"] == "completed", result["errors"]
    assert result["status"] == "incomplete"  # Missing collectors never become a pass.
    assert seen == [f"product-{i}" for i in range(5)] * 2
    for epoch in result["epochs"]:
        assert epoch["complete"]
        assert epoch["metrics"]["real_samples"] == 5
        assert epoch["metrics"]["batches"] == 2
        if training:
            assert epoch["metrics"]["optimizer_updates"] == 2
    if training:
        assert models[0].weight.item() > 0.25
    coverage = json.loads((tmp_path / "run/coverage.json").read_text())
    assert coverage["metrics"]["materialization.elapsed_ns"]["status"] == "complete"
