// The upstream hash (docs/tuning_system.md, "The tuning file format"): FNV-1a 64 over the
// canonical text of a section's resolved measured values, `key=value` lines sorted by key, with
// integers in decimal and floats in shortest round-trip form. Shared by build.rs and the crate.

pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// The canonical text of `(key, value text)` pairs: sorted by key, one `key=value` line each.
pub fn canonical(pairs: &[(&str, String)]) -> String {
    let mut sorted: Vec<&(&str, String)> = pairs.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    for (k, v) in sorted {
        out.push_str(k);
        out.push('=');
        out.push_str(v);
        out.push('\n');
    }
    out
}

pub fn hash_pairs(pairs: &[(&str, String)]) -> u64 {
    fnv1a(canonical(pairs).as_bytes())
}

/// A table knob's identity for `RESOLVED_BITS` and the accessor-agreement test: FNV-1a 64 over its
/// numbers' little-endian bytes.
pub fn table_bits(t: &[u32]) -> u64 {
    let mut h = FNV_OFFSET;
    for v in t {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}
