use crate::builder::EdgeWriter;
use crate::builder::{EdgeRec, IndexBuildError};
use crate::layout::RelationLayout;

/// Truncate per-relation edge lists for a node to keep the offset delta within u16::MAX.
/// Strategy: identify the longest edge lists and "level" them down together so they end
/// up the same length (and still longer than all other lists), repeating until the node
/// payload fits in the allowed delta.
pub(crate) fn truncate_counts(
    buckets: &mut [Vec<EdgeRec>],
    counts: &mut [usize],
    layouts: &[RelationLayout],
    current_offset: u64,
    prev_offset: u64,
    extra_bytes: u64,
) -> Result<(), IndexBuildError> {
    let base_span = current_offset
        .checked_sub(prev_offset)
        .ok_or(IndexBuildError::CountOverflow)?;
    let budget = (u16::MAX as u64)
        .checked_sub(base_span)
        .ok_or(IndexBuildError::OffsetDeltaOverflow)?;

    let entry_bytes: Vec<u64> = layouts.iter().map(|l| l.entry_bytes as u64).collect();

    loop {
        let node_bytes = EdgeWriter::node_length_bytes(layouts, counts)?
            .checked_add(extra_bytes)
            .ok_or(IndexBuildError::CountOverflow)?;
        if node_bytes <= budget {
            // Apply truncations to buckets and propagate counts.
            for (idx, target) in counts.iter().enumerate() {
                buckets[idx].truncate(*target);
            }
            return Ok(());
        }

        // Items sorted by current length desc.
        let mut items: Vec<(usize, usize)> = counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| (i, c))
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));

        if items.is_empty() {
            return Err(IndexBuildError::OffsetDeltaOverflow);
        }

        // Identify the top "plateau" (all relations with the current max length).
        let top_len = items[0].1;
        let mut level = 1;
        while level < items.len() && items[level].1 == top_len {
            level += 1;
        }
        let next_len = items.get(level).map(|p| p.1).unwrap_or(0);
        let max_drop = top_len.saturating_sub(next_len).max(1);

        // Estimate how many elements we need to drop from this plateau to reach budget,
        // using entry byte size as the dominating term (varint prefix is small).
        let mut bytes_per_step: u64 = 0;
        for i in 0..level {
            bytes_per_step = bytes_per_step.saturating_add(entry_bytes[items[i].0]);
        }
        if bytes_per_step == 0 {
            return Err(IndexBuildError::OffsetDeltaOverflow);
        }
        let needed = node_bytes
            .checked_sub(budget)
            .ok_or(IndexBuildError::OffsetDeltaOverflow)?;
        let steps_needed = (needed + bytes_per_step - 1) / bytes_per_step;
        let drop = max_drop.min(steps_needed as usize).max(1);

        for i in 0..level {
            counts[items[i].0] = counts[items[i].0].saturating_sub(drop);
        }
    }
}

// helper removed; count varint sizing is handled in builder hot path now.
