use crate::Hash;

fn hash_byte<H: Hash>(hash: H, pass: usize) -> u8 {
    let word_bytes = std::mem::size_of::<H>();
    if word_bytes == 8 {
        let value = hash.low();
        ((value >> (pass * 8)) & 0xFF) as u8
    } else {
        let value = if pass < 8 { hash.low() } else { hash.high() };
        let shift = (pass % 8) * 8;
        ((value >> shift) & 0xFF) as u8
    }
}

pub(crate) fn radix_sort_hashes<H: Hash>(hashes: &mut [H], tmp: &mut Vec<H>) {
    if hashes.len() <= 1 {
        return;
    }
    let passes = std::mem::size_of::<H>();
    tmp.resize(hashes.len(), H::default());

    let mut use_tmp = false;

    for pass in 0..passes {
        let (input, output): (&[H], &mut [H]) = if use_tmp {
            (tmp.as_slice(), hashes)
        } else {
            (hashes, tmp.as_mut_slice())
        };
        let mut counts = [0usize; 256];
        for &hash in input.iter() {
            counts[hash_byte(hash, pass) as usize] += 1;
        }
        let mut total = 0usize;
        for count in counts.iter_mut() {
            let current = *count;
            *count = total;
            total = total.saturating_add(current);
        }
        for &hash in input.iter() {
            let bucket = hash_byte(hash, pass) as usize;
            let idx = counts[bucket];
            output[idx] = hash;
            counts[bucket] = idx + 1;
        }
        use_tmp = !use_tmp;
    }

    if use_tmp {
        hashes.copy_from_slice(tmp.as_slice());
    }
}
