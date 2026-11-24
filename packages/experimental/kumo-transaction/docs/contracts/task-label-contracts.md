# Task label contracts

Keep event, entity, and sequence labels distinct. Fraud labels are
event-level with `available_at`. A sequence label is 1 when **any** event in
the sequence has a positive label that is already available at the cutoff,
and 0 otherwise (`tfm.data.collate.sequence_label`). With a label window
([ADR 0013](../architecture/0013-fraud-probe-split-integrity.md)), only events
inside that side's window count, so an earlier period's fraud cannot make a
row positive.
