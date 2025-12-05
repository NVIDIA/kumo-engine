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
) -> Result<(), IndexBuildError> {
    let base_span = current_offset
        .checked_sub(prev_offset)
        .ok_or(IndexBuildError::CountOverflow)?;
    let budget = (u16::MAX as u64)
        .checked_sub(base_span)
        .ok_or(IndexBuildError::OffsetDeltaOverflow)?;

    let mut working_counts = counts.to_vec();

    loop {
        let node_bytes = EdgeWriter::node_length_bytes(layouts, &working_counts)?;
        if node_bytes <= budget {
            // Apply truncations to buckets and propagate counts.
            for (idx, target) in working_counts.iter().enumerate() {
                buckets[idx].truncate(*target);
            }
            counts.copy_from_slice(&working_counts);
            return Ok(());
        }

        // Items sorted by current length desc.
        let mut items: Vec<(usize, usize)> = working_counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| (i, c))
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));

        if items.is_empty() {
            return Err(IndexBuildError::OffsetDeltaOverflow);
        }

        let mut leveled = false;
        for level in 0..items.len() {
            let current_len = items[level].1;
            let next_len = items.get(level + 1).map(|p| p.1).unwrap_or(0);

            // If there is no change in length at this level, skip ahead to avoid
            // getting stuck when multiple relations have identical lengths.
            if current_len == next_len {
                continue;
            }

            // Level the top (level+1) lists down to next_len and see if it fits.
            let mut candidate_counts = working_counts.clone();
            for i in 0..=level {
                candidate_counts[items[i].0] = next_len;
            }
            if candidate_counts == working_counts {
                continue;
            }
            let candidate_bytes = EdgeWriter::node_length_bytes(layouts, &candidate_counts)?;
            if candidate_bytes > budget {
                // Still too large; accept this leveling and iterate again.
                working_counts = candidate_counts;
                leveled = true;
                break;
            }

            // Binary search for the largest length between next_len and current_len that fits.
            let mut low = next_len;
            let mut high = current_len;
            let mut best = next_len;
            while low <= high {
                let mid = (low + high) / 2;
                let mut trial = working_counts.clone();
                for i in 0..=level {
                    trial[items[i].0] = mid;
                }
                let trial_bytes = EdgeWriter::node_length_bytes(layouts, &trial)?;
                if trial_bytes <= budget {
                    best = mid;
                    if mid == current_len {
                        break;
                    }
                    low = mid + 1;
                } else {
                    if mid == 0 {
                        break;
                    }
                    high = mid - 1;
                }
            }
            for i in 0..=level {
                working_counts[items[i].0] = best;
            }
            leveled = true;
            break;
        }

        if !leveled {
            // Every list already zero or could not be reduced further.
            return Err(IndexBuildError::OffsetDeltaOverflow);
        }
    }
}

pub(crate) fn varint_len(v: u32) -> usize {
    if v < (1 << 7) {
        1
    } else if v < (1 << 14) {
        2
    } else if v < (1 << 21) {
        3
    } else if v < (1 << 28) {
        4
    } else {
        5
    }
}
