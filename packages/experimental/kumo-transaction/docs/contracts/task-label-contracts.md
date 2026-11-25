# Task label contracts

Keep event, entity, and sequence labels distinct. Fraud labels are
event-level with `available_at`. A sequence label is 1 when **any** event in
the sequence has a positive label that is already available at the cutoff,
and 0 otherwise (`tfm.data.collate.sequence_label`). With a label window
([probe split rules](../concepts/evaluation-protocol.md)), only events
inside that side's window count, so an earlier period's fraud cannot make a
row positive.
