"""Public Kumo API adapter, independent of source schema and model architecture."""

from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interface import Workload, _nothing


@dataclass
class KumoInputs:
    contract: Any
    dataset: Any
    engine: Any
    fitting_anchors: Any
    epoch_anchors: Any
    materialize_options: Mapping[str, Any] = field(default_factory=dict)
    batch_options: Mapping[str, Any] = field(default_factory=dict)
    reference: bool = False


def kumo_workload(
    *,
    identity,
    inputs,
    count,
    consumer_factory=None,
    synchronize=_nothing,
    inapplicable=None,
    device="cpu",
    load_fitted_state=None,
    save_fitted_state=None,
    probe_reopen=False,
):
    """Build a Workload using any Kumo contract/dataset/backend and model.

    ``inputs(recorder) -> KumoInputs`` prepares sources/anchors and records actual
    input preparation. ``epoch_anchors`` is a reusable anchor table/request
    collection, not a single-use generator. ``consumer_factory(task, recorder)``
    optionally returns ``step(batch, recorder) -> bool`` with its own optimizer.
    Each factory call must create fresh mutable engine/model/optimizer state.

    Enclosing prepare/materialize spans and scoped internal observers retain
    separate ownership. Unsupported internal boundaries remain incomplete.
    """
    stack = ExitStack()
    state = {}

    def prepare(recorder):
        from kumo.pytorch.api import Artifact
        from kumo.pytorch.api import prepare as prepare_task

        from .kumo_observers import KumoObservers

        observers = stack.enter_context(KumoObservers(recorder))
        with recorder.span("ingestion"):
            selected = inputs(recorder)
            if not isinstance(selected, KumoInputs):
                raise TypeError("inputs must return KumoInputs")
            if "epoch" in selected.batch_options:
                raise ValueError("The benchmark runner owns epoch selection")
            state["inputs"] = selected
            if selected.reference:
                observers.patch(
                    selected.engine, "open", observers.timed("reference.load")
                )
            with recorder.span("task.setup"):
                plan = stack.enter_context(
                    prepare_task(selected.contract, selected.dataset, selected.engine)
                )
        observers.attach_source(plan._source)
        if probe_reopen:
            plan.close()
            plan = stack.enter_context(
                prepare_task(selected.contract, selected.dataset, selected.engine)
            )
            observers.attach_source(plan._source)
        with (
            recorder.span("materialization"),
            recorder.span("reference.fit") if selected.reference else nullcontext(),
        ):
            options = dict(selected.materialize_options)
            loaded = options.get("fitted")
            if load_fitted_state is not None:
                if "fitted" in options:
                    raise ValueError("Choose one fitted-state source")
                with recorder.span("state.load"):
                    loaded = Artifact.load(load_fitted_state).fitted_state
                options.pop("fit_split", None)
                options["fitted"] = loaded
            task = stack.enter_context(
                plan.materialize(selected.fitting_anchors, **options)
            )
            if (
                loaded is not None
                and loaded.has_targets
                and task.fitted_state.fingerprint != loaded.fingerprint
            ):
                raise ValueError("Reused fitted state changed during materialization")
        if save_fitted_state is not None:
            Path(save_fitted_state).mkdir(parents=True, exist_ok=False)
            task.to_artifact().save(save_fitted_state)
        recorder.write(
            "fitting_state",
            mode="fresh"
            if loaded is None
            else "reused"
            if loaded.has_targets
            else "partial_reuse",
            fingerprint=task.fitted_state.fingerprint,
            contract_fingerprint=selected.contract.fingerprint,
            anchors_fingerprint=task.fingerprints["anchors"],
            loaded_from=str(load_fitted_state)
            if load_fitted_state is not None
            else None,
            saved_to=str(save_fitted_state) if save_fitted_state is not None else None,
        )
        state["task"] = task
        state["recorder"] = recorder
        observers.attach_source(task._assembler._source)
        if consumer_factory is not None:
            state["step"] = consumer_factory(task, recorder)
            if not callable(state["step"]):
                raise TypeError("consumer_factory must return a callable training step")

    def batches(epoch):
        selected = state["inputs"]
        recorder = state["recorder"]
        with recorder.span("reference.plan") if selected.reference else nullcontext():
            iterator = iter(
                state["task"].batches(
                    selected.epoch_anchors, epoch=epoch, **selected.batch_options
                )
            )
        while True:
            with (
                recorder.span("reference.batch")
                if selected.reference
                else nullcontext()
            ):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            yield batch

    def step(batch, recorder):
        return state["step"](batch, recorder)

    return Workload(
        identity=identity,
        prepare=prepare,
        batches=batches,
        count=count,
        step=step if consumer_factory is not None else None,
        synchronize=synchronize,
        close=stack.close,
        inapplicable=inapplicable or {},
        device=device,
    )
