# Task label contracts

Keep event, entity, and sequence labels distinct. Fraud labels are
event-level with `available_at`. A sequence label is 1 when **any** event in
the sequence has a positive label that is already available at the cutoff,
and 0 otherwise (`tfm.data.collate.sequence_label`). With a label window
(`LabelWindow`, defined below), only events
inside that side's window count, so an earlier period's fraud cannot make a
row positive.

## Probe-side label windows

`tfm.data.labels.windows.LabelWindow` explicitly declares the event-time interval
`(start, end]`: start is exclusive and end is inclusive. `start=None` means no
lower bound. Event times and window endpoints are compared in UTC. A label must
also be observable at the sequence cutoff; membership in a label window never
bypasses its separate availability clock.

For chronological train, validation and test cutoffs `t_train`, `t_validation`
and `t_test`, `side_label_windows` selects:

| Side | Event times whose labels may count |
| --- | --- |
| Train | `(t_train - observation_window, t_train]` |
| Validation | `(t_train, t_validation]` |
| Test | `(t_validation, t_test]` |

Older events can remain input history. Their labels do not count on the later
side. Windows are explicit and recorded in the split manifest; they are not
inferred from a sequence's length or oldest event. Entity-level forecast labels
with their own prediction timestamps and horizons remain a separate task
contract, not event-level fraud labels.
