"""Private checks and tensor operations for PRAGMA's native Kumo interface."""

from collections.abc import Mapping, Sequence

import torch


SPECIALS = ("[PAD]", "[UNK]", "[MASK]", "[USR]", "[EVT]")


def mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def tensor(value, name, dtype=None, *, cpu=False, device=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if cpu and value.device.type != "cpu":
        raise TypeError(f"{name} must be a CPU tensor; validate before transfer")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on the same device as the model")
    return value


def identifiers(value, name, *, cpu=False, check_values=True):
    if isinstance(value, torch.Tensor):
        tensor(value, name, torch.long, cpu=cpu)
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = value.tolist() if check_values else None
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise TypeError(f"{name} must be a sequence of identifiers")
    if check_values and any(type(item) not in (str, int) or item == "" for item in values):
        raise ValueError(f"{name} must contain nonempty strings or integer identifiers")
    return tuple(values) if check_values else value


def remap(ids):
    """Swap reserved UNK/MASK IDs, preserving all fitted ordinary token IDs."""
    return torch.where(ids == 1, 2, torch.where(ids == 2, 1, ids))


def offsets(lengths):
    return torch.cat((lengths.new_zeros(1), lengths.long().cumsum(0))).to(torch.int32)


def check_ids(ids, valid, size, name):
    observed = ids[valid]
    if ((observed < 0) | (observed >= size)).any().item():
        raise ValueError(f"{name} contains an ID outside the fitted vocabulary")


def right_padding(valid, name):
    if valid.ndim != 2 or (valid[:, 1:] & ~valid[:, :-1]).any().item():
        raise ValueError(f"{name} must be a two-dimensional right-padding mask")


def history_fields(batch, history_input, *, check_values=False, cpu=False, device=None):
    """Read typed Kumo fields and aligned derived inputs without changing the batch."""
    history = mapping(batch.inputs[history_input], history_input)
    sequence = mapping(history["sequence"], "history.sequence")
    timing = mapping(batch.inputs[history_input + "_timing"], "history timing")
    calendar = mapping(batch.inputs[history_input + "_calendar"], "history calendar")
    valid = sequence["valid"]
    valid = tensor(valid, "history.sequence.valid", torch.bool, cpu=cpu, device=device)
    fields = [("age_seconds", timing["age_seconds"]),
              *((name, calendar[name]) for name in ("hour", "weekday", "day"))]
    values = []
    for name, field in fields:
        value = tensor(field["values"], name + ".values", cpu=cpu, device=device)
        mask = tensor(field["valid"], name + ".valid", torch.bool, cpu=cpu, device=device)
        if value.shape != valid.shape or mask.shape != valid.shape:
            raise ValueError("derived history fields must align with the event sequence")
        if check_values and not torch.equal(mask, valid):
            raise ValueError("derived history validity must match the event sequence")
        values.append(value)
    elapsed, *columns = values
    return {**history,
            "sequence": {**sequence, "elapsed_seconds": elapsed},
            "calendar": torch.stack([column[valid] for column in columns], dim=-1)}


def inspect(batch, model, *, check_values, cpu=False, device=None):
    """Check public batch fields without modifying or replacing any input."""
    metadata, inputs = mapping(batch.metadata, "batch.metadata"), mapping(batch.inputs, "batch.inputs")
    entities = identifiers(metadata["entity_ids"], "metadata.entity_ids", cpu=cpu, check_values=check_values)
    samples = identifiers(metadata["sample_ids"], "metadata.sample_ids", cpu=cpu, check_values=check_values)
    size = len(entities)
    if not size or len(samples) != size:
        raise ValueError("metadata requires one sample ID per entity and a nonempty batch")
    if "is_padding" in metadata:
        padding = tensor(metadata["is_padding"], "metadata.is_padding", torch.bool, cpu=cpu, device=device)
        if padding.shape != (size,):
            raise ValueError("metadata.is_padding must contain one flag per request")
    history = history_fields(batch, model.history_input, check_values=check_values, cpu=cpu, device=device)
    profile = None if model.profile_input is None else mapping(inputs[model.profile_input], model.profile_input)
    keys, values, positions, valid = [tensor(history[key], "history." + key, dtype, cpu=cpu, device=device)
        for key, dtype in (("field_ids", torch.long), ("value_ids", torch.long),
                           ("position_ids", torch.long), ("valid", torch.bool))]
    if keys.ndim != 1 or any(value.shape != keys.shape for value in (values, positions, valid)):
        raise ValueError("history fields must be aligned packed arrays without padding")
    boundaries = tensor(history["field_offsets"], "history.field_offsets", cpu=cpu, device=device)
    if boundaries.dtype not in (torch.int32, torch.int64) or boundaries.ndim != 1 or not boundaries.numel():
        raise ValueError("field offsets must partition the complete packed token array")
    event_count = boundaries.numel() - 1
    if max(keys.numel(), event_count) > torch.iinfo(torch.int32).max:
        raise ValueError("packed offsets exceed the PRAGMA int32 boundary limit")
    sequence = mapping(history["sequence"], "history.sequence")
    event_indices = tensor(sequence["event_indices"], "sequence.event_indices", torch.long, cpu=cpu, device=device)
    seq_valid = tensor(sequence["valid"], "sequence.valid", torch.bool, cpu=cpu, device=device)
    elapsed = tensor(sequence["elapsed_seconds"], "sequence.elapsed_seconds", cpu=cpu, device=device)
    if event_indices.ndim != 2 or event_indices.shape[0] != size or seq_valid.shape != event_indices.shape or elapsed.shape != event_indices.shape:
        raise ValueError("event sequence must contain aligned [B,Emax] arrays")
    owners = tensor(history["owner"], "history.owner", torch.long, cpu=cpu, device=device)
    history_positions = tensor(history["history_position"], "history.history_position", torch.long, cpu=cpu, device=device)
    if owners.shape != (event_count,) or history_positions.shape != (event_count,):
        raise ValueError("event owners/positions must agree with sequence boundaries")
    calendar = tensor(history["calendar"], "history.calendar", cpu=cpu, device=device)
    if calendar.shape != (event_count, 3):
        raise ValueError("calendar must have integral hour,weekday,day columns")
    if check_values:
        if not valid.all().item():
            raise ValueError("history fields must be aligned packed arrays without padding")
        check_ids(keys, valid, model.config.vocab_size, "history.key_ids")
        check_ids(values, valid, model.config.vocab_size, "history.value_ids")
        if (positions < 0).any().item():
            raise ValueError("within-field positions cannot be negative")
        if boundaries[0].item() != 0 or boundaries[-1].item() != keys.numel() or (boundaries[1:] < boundaries[:-1]).any().item():
            raise ValueError("field offsets must partition the complete packed token array")
        right_padding(seq_valid, "sequence.valid")
        if not torch.equal(event_indices[seq_valid], torch.arange(event_count, device=keys.device)):
            raise ValueError("event sequence must cover packed events once in request order")
        elapsed_flat = elapsed[seq_valid].float()
        if not torch.isfinite(elapsed_flat).all().item() or (elapsed_flat < 0).any().item():
            raise ValueError("event elapsed seconds must be finite and nonnegative")
        if ((elapsed[:, 1:] > elapsed[:, :-1]) & seq_valid[:, 1:]).any().item():
            raise ValueError("events must be chronologically ordered within each request")
        counts = seq_valid.sum(1)
        if model.elapsed_from == "latest_event":
            for row, count in enumerate(counts.tolist()):
                if count and elapsed[row, count - 1].item() != 0:
                    raise ValueError("latest_event elapsed reference must end at zero")
        expected_owners = torch.repeat_interleave(torch.arange(size, device=keys.device), counts, output_size=event_count)
        expected_positions = torch.arange(event_count, device=keys.device) - torch.repeat_interleave(offsets(counts)[:-1].long(), counts, output_size=event_count)
        if not torch.equal(owners, expected_owners) or not torch.equal(history_positions, expected_positions):
            raise ValueError("event owners/positions must agree with sequence boundaries")
        if not torch.isfinite(calendar).all().item() or not torch.equal(calendar, calendar.long().to(calendar.dtype)):
            raise ValueError("calendar must have integral hour,weekday,day columns")
        for col, low, high in ((0, 0, 23), (1, 0, 6), (2, 1, 31)):
            if ((calendar[:, col] < low) | (calendar[:, col] > high)).any().item():
                raise ValueError("calendar value is out of range")
    if profile is not None:
        inspect_profile(profile, size, model.config.vocab_size, check_values=check_values, cpu=cpu, device=device)
    return history, profile, size


def inspect_profile(raw, size, vocabulary_size, *, check_values, cpu=False, device=None):
    keys, values, positions, valid, elapsed = [tensor(raw[key], "profile." + key, dtype, cpu=cpu, device=device)
        for key, dtype in (("field_ids", torch.long), ("value_ids", torch.long),
                           ("position_ids", torch.long), ("valid", torch.bool), ("elapsed_seconds", None))]
    if keys.ndim != 2 or keys.shape[0] != size or any(value.shape != keys.shape for value in (values, positions, valid, elapsed)):
        raise ValueError("profile fields must have aligned [B,P] shapes")
    if check_values:
        right_padding(valid, "profile.valid")
        check_ids(keys, valid, vocabulary_size, "profile.key_ids")
        check_ids(values, valid, vocabulary_size, "profile.value_ids")
        if (positions[valid] < 0).any().item() or not torch.isfinite(elapsed[valid]).all().item() or (elapsed[valid] < 0).any().item():
            raise ValueError("profile positions and elapsed seconds must be finite/nonnegative")
        previous = torch.zeros_like(valid)
        if valid.shape[1]:
            previous[:, 1:] = valid[:, :-1]
        continued = valid & (positions != 0)
        valid_piece = torch.zeros_like(valid)
        if valid.shape[1]:
            valid_piece[:, 1:] = ((positions[:, 1:] == positions[:, :-1] + 1)
                                  & (keys[:, 1:] == keys[:, :-1]) & (elapsed[:, 1:] == elapsed[:, :-1]))
        if (continued & ~(previous & valid_piece)).any().item():
            raise ValueError("profile field pieces must start at zero and retain key/time")
        if valid.sum().item() > torch.iinfo(torch.int32).max:
            raise ValueError("packed offsets exceed the PRAGMA int32 boundary limit")


def token_events(history):
    lengths = history["field_offsets"][1:].long() - history["field_offsets"][:-1].long()
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=lengths.device), lengths,
                                   output_size=history["field_ids"].numel())


def selected_targets(batch, history_input, target_name):
    history, target = batch.inputs[history_input], batch.targets[target_name]
    owners = history["owner"][token_events(history)]
    valid = target["valid"] & (target["values"] != -100)
    if "is_padding" in batch.metadata:
        valid = valid & ~batch.metadata["is_padding"][owners]
    selected = valid.nonzero(as_tuple=False).flatten()
    return selected, remap(target["values"][selected])


def inspect_targets(batch, model, target_name, *, cpu=True):
    targets = mapping(batch.targets, "batch.targets")
    if target_name is None:
        if "masked_values" in targets:
            raise ValueError("supervised PRAGMA inputs must omit MaskedValuesTarget")
        return
    raw = mapping(targets[target_name], target_name)
    labels = tensor(raw["values"], "masked.values", torch.long, cpu=cpu)
    valid = tensor(raw["valid"], "masked.valid", torch.bool, cpu=cpu)
    history = batch.inputs[model.history_input]
    if labels.shape != history["field_ids"].shape or valid.shape != labels.shape:
        raise ValueError("masked targets must align with packed history fields")
    if "offsets" in raw and not torch.equal(tensor(raw["offsets"], "masked.offsets", cpu=cpu).long(), history["field_offsets"].long()):
        raise ValueError("masked target offsets must match history field offsets")
    selected, _ = selected_targets(batch, model.history_input, target_name)
    observed = labels[selected]
    if ((observed < 0) | (observed >= model.config.vocab_size)).any().item():
        raise ValueError("masked.values contains an ID outside the fitted vocabulary")
