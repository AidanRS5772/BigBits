//! Dispatchers' small-input tables: change points measured by the tuner (`tuner/tables.rs`), read
//! where a dispatcher's equations would cost more than the rule's share of the call. Lookups are
//! linear scans: a table holds a few change points (one input) or a few per row (two).

/// A one-input table `[size, alg, size, alg, …]`: the algorithm of the last change point at or
/// below `n` (the first one's for smaller n).
#[inline(always)]
pub fn pick1(t: &[u32], n: u64) -> usize {
    debug_assert!(t.len() >= 2 && t.len() % 2 == 0);
    let mut alg = t[1];
    let mut i = 2;
    while i < t.len() && t[i] as u64 <= n {
        alg = t[i + 1];
        i += 2;
    }
    alg as usize
}

/// A table with its row index: what a table accessor returns. `rows` is empty for a one-input
/// table. For a two-input table `[s, l, alg, …]`, `rows[s]` is the entry starting the row with the
/// largest s at or below `s` (u32::MAX below the first row), for every s up to the last row's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table2 {
    pub t: &'static [u32],
    pub rows: &'static [u32],
}

/// The row index of a two-input table (see `Table2`).
pub fn row_index(t: &[u32]) -> Vec<u32> {
    let last = t[t.len() - 3] as usize;
    let mut rows = vec![u32::MAX; last + 1];
    let mut i = 0;
    while i < t.len() / 3 {
        let s = t[3 * i] as usize;
        if i == 0 || t[3 * (i - 1)] as usize != s {
            for r in rows[s..].iter_mut() {
                *r = i as u32;
            }
        }
        i += 1;
    }
    rows
}

impl Table2 {
    /// In the row with the largest s at or below `s`, the algorithm of the last change point at
    /// or below `l` (the row's first for smaller l, and the first row's first below every row).
    /// One indexed read finds the row; rows hold a few change points.
    #[inline(always)]
    pub fn pick(&self, s: usize, l: usize) -> usize {
        let (t, rows) = (self.t, self.rows);
        let r = rows[s.min(rows.len() - 1)];
        if r == u32::MAX {
            return t[2] as usize;
        }
        let mut i = 3 * r as usize;
        let row = t[i];
        while i + 5 < t.len() && t[i + 3] == row && t[i + 4] as usize <= l {
            i += 3;
        }
        t[i + 2] as usize
    }
}
