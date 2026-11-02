// Compile-time plans for theta's powers q^(n²), n = 1..=2T + 1 (trans.rs). Value 0 is q, and op v
// makes value v = a b from two earlier values, a square when a == b. PARALLEL takes the shortest
// dependency path (earliest finish, squares 2 and products 3), and SERIAL the least work, each op
// weighted by the window of the first term it feeds: a greedy plan, or PARALLEL where that's less.

pub(crate) const THETA_MAX_TERMS: usize = 16;
pub(crate) const MAX_VALUES: usize = 64;
const MAX_K: usize = 2 * THETA_MAX_TERMS + 1;
const MAX_EXP: usize = MAX_K * MAX_K;
const NONE: u8 = u8::MAX;
const SQR: u64 = 2;
const MUL: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ThetaOp {
    pub a: u8,
    pub b: u8,
    // n when the value is q^(n²), else 0.
    pub term: u8,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ThetaPlan {
    pub op: [ThetaOp; MAX_VALUES],
    pub exp: [u16; MAX_VALUES],
    pub values: usize,
    // Slots for running the ops in order: a slot frees after its value's last reader, or once
    // it's summed if nothing reads it. slots is that order's live peak.
    pub slot: [u8; MAX_VALUES],
    pub slots: usize,
    // The most ops sharing a start time, which caps useful workers.
    pub width: usize,
}

pub(crate) struct ThetaPlans<const T: usize>;

impl<const T: usize> ThetaPlans<T> {
    const BOTH: (ThetaPlan, ThetaPlan) = plans(T);
    pub(crate) const SERIAL: ThetaPlan = Self::BOTH.0;
    pub(crate) const PARALLEL: ThetaPlan = Self::BOTH.1;
}

const fn plans(t: usize) -> (ThetaPlan, ThetaPlan) {
    let (g, p) = (color(serial_plan(t)), color(parallel_plan(t)));
    (if work(&p, t) < work(&g, t) { p } else { g }, p)
}

// W(n) = 4(T + 1)² - (n² - [n odd]), term n's window in units of 1 / 4(T + 1)² of the frame. It
// falls with n, so a value is as wide as the first term it feeds.
const fn window(t: usize, n: usize) -> u64 {
    4 * (t as u64 + 1) * (t as u64 + 1) - (n * n - (n & 1)) as u64
}

// Squares 2, products 3, each times its value's width.
const fn work(p: &ThetaPlan, t: usize) -> u64 {
    let mut first = [u8::MAX; MAX_VALUES];
    let mut v = p.values;
    while v > 0 {
        v -= 1;
        let o = p.op[v];
        if o.term != 0 && o.term < first[v] {
            first[v] = o.term;
        }
        if v > 0 {
            if first[v] < first[o.a as usize] {
                first[o.a as usize] = first[v];
            }
            if first[v] < first[o.b as usize] {
                first[o.b as usize] = first[v];
            }
        }
    }
    let mut sum = 0;
    v = 1;
    while v < p.values {
        let o = p.op[v];
        sum += op_cost((o.a as usize, o.b as usize)) * window(t, first[v] as usize);
        v += 1;
    }
    sum
}

const fn min(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

const fn op_cost((a, b): (usize, usize)) -> u64 {
    if a == b {
        SQR
    } else {
        MUL
    }
}

struct Builder {
    p: ThetaPlan,
    // The value with each exponent, and the exponents present in ascending order.
    at: [u8; MAX_EXP + 1],
    sorted: [u16; MAX_VALUES],
}

impl Builder {
    const fn new(t: usize) -> Self {
        assert!(
            t <= THETA_MAX_TERMS,
            "theta's terms are capped at THETA_MAX_TERMS"
        );
        let mut b = Builder {
            p: ThetaPlan {
                op: [ThetaOp {
                    a: 0,
                    b: 0,
                    term: 0,
                }; MAX_VALUES],
                exp: [0; MAX_VALUES],
                values: 1,
                slot: [0; MAX_VALUES],
                slots: 0,
                width: 1,
            },
            at: [NONE; MAX_EXP + 1],
            sorted: [0; MAX_VALUES],
        };
        b.p.op[0].term = 1;
        b.p.exp[0] = 1;
        b.at[1] = 0;
        b.sorted[0] = 1;
        b
    }

    const fn has(&self, e: usize) -> bool {
        e <= MAX_EXP && self.at[e] != NONE
    }

    // Value e = a + b.
    const fn push(&mut self, e: usize, a: usize, b: usize) {
        let v = self.p.values;
        assert!(
            v < MAX_VALUES,
            "a theta plan holds at most MAX_VALUES values"
        );
        let mut n = 1;
        while n * n < e {
            n += 1;
        }
        let term = if n * n == e { n as u8 } else { 0 };
        self.p.op[v] = ThetaOp {
            a: self.at[a],
            b: self.at[b],
            term,
        };
        self.p.exp[v] = e as u16;
        self.at[e] = v as u8;
        let mut i = v;
        while i > 0 && self.sorted[i - 1] as usize > e {
            self.sorted[i] = self.sorted[i - 1];
            i -= 1;
        }
        self.sorted[i] = e as u16;
        self.p.values = v + 1;
    }

    // e from one op on present values: a square when e / 2 is present, else the smallest a.
    // (0, 0) when there's none.
    const fn make1(&self, e: usize) -> (usize, usize) {
        if e % 2 == 0 && self.has(e / 2) {
            return (e / 2, e / 2);
        }
        let mut i = 0;
        while i < self.p.values {
            let a = self.sorted[i] as usize;
            if a > e - a {
                break;
            }
            if self.has(e - a) {
                return (a, e - a);
            }
            i += 1;
        }
        (0, 0)
    }
}

// Targets in order, each by its cheapest one op, else one helper h (from present values) and
// e = h + (e - h). An op made for term n is as wide as n's window, since later terms are narrower.
const fn serial_plan(t: usize) -> ThetaPlan {
    let mut b = Builder::new(t);
    let k = 2 * t + 1;
    // The target each helper exponent was last tried for.
    let mut seen = [0u8; MAX_EXP + 1];
    let mut n = 2;
    while n <= k {
        let e = n * n;
        let one = b.make1(e);
        if one.0 != 0 {
            b.push(e, one.0, one.1);
            n += 1;
            continue;
        }
        // One op never costs more than 3 W, two at least 4 W, so helpers only matter here.
        let w = window(t, n);
        let (mut best, mut bh, mut hop) = (u64::MAX, 0, (0, 0));
        let vals = b.p.values;
        let mut i = 0;
        while i < vals {
            let mut j = i;
            while j < vals {
                let h = b.sorted[i] as usize + b.sorted[j] as usize;
                if h < e && !b.has(h) && seen[h] != n as u8 {
                    seen[h] = n as u8;
                    let r = e - h;
                    if b.has(r) || r == h {
                        let o = b.make1(h);
                        let cost = w * (op_cost(o) + op_cost((h, r)));
                        if cost < best {
                            (best, bh, hop) = (cost, h, o);
                        }
                    }
                }
                j += 1;
            }
            i += 1;
        }
        assert!(best != u64::MAX, "no one- or two-op step for a theta term");
        b.push(bh, hop.0, hop.1);
        let r = e - bh;
        b.push(e, min(bh, r), bh + r - min(bh, r));
        n += 1;
    }
    b.p
}

// The user's earliest-finish schedule: finish[m] = min over a + b = m of max(finish[a], finish[b])
// + (2 for a square, else 3), keeping the first minimum, then only what the targets need, in
// order of start time and exponent.
const fn parallel_plan(t: usize) -> ThetaPlan {
    let mut b = Builder::new(t);
    let k = 2 * t + 1;
    let top = k * k;
    let mut fin = [0u16; MAX_EXP + 1];
    let mut par = [0u16; MAX_EXP + 1];
    let mut m = 2;
    while m <= top {
        let (mut best, mut ba) = (u16::MAX, 0);
        let mut a = 1;
        while a <= m / 2 {
            let (fa, fb) = (fin[a], fin[m - a]);
            let c = (if fa > fb { fa } else { fb }) + if a == m - a { 2 } else { 3 };
            if c < best {
                (best, ba) = (c, a);
            }
            a += 1;
        }
        (fin[m], par[m]) = (best, ba as u16);
        m += 1;
    }
    let mut need = [false; MAX_EXP + 1];
    let mut n = 1;
    while n <= k {
        need[n * n] = true;
        n += 1;
    }
    let mut start = [0u16; MAX_EXP + 1];
    m = top;
    while m >= 2 {
        if need[m] {
            let a = par[m] as usize;
            (need[a], need[m - a]) = (true, true);
            let (fa, fb) = (fin[a], fin[m - a]);
            start[m] = if fa > fb { fa } else { fb };
        }
        m -= 1;
    }
    // Stable insertion by start, so ties keep exponent order.
    let mut order = [0u16; MAX_VALUES];
    let mut cnt = 0;
    m = 2;
    while m <= top {
        if need[m] {
            assert!(
                cnt + 1 < MAX_VALUES,
                "a theta plan holds at most MAX_VALUES values"
            );
            let mut i = cnt;
            while i > 0 && start[order[i - 1] as usize] > start[m] {
                order[i] = order[i - 1];
                i -= 1;
            }
            order[i] = m as u16;
            cnt += 1;
        }
        m += 1;
    }
    let (mut i, mut run) = (0, 0);
    while i < cnt {
        let m = order[i] as usize;
        let a = par[m] as usize;
        b.push(m, a, m - a);
        run = if i > 0 && start[order[i - 1] as usize] == start[m] {
            run + 1
        } else {
            1
        };
        if run > b.p.width {
            b.p.width = run;
        }
        i += 1;
    }
    b.p
}

// In-order slots: each value takes a free slot (its operands still hold theirs), then frees the
// operands it read last, and itself if nothing reads it.
const fn color(mut p: ThetaPlan) -> ThetaPlan {
    let mut last = [NONE; MAX_VALUES];
    let mut v = 1;
    while v < p.values {
        last[p.op[v].a as usize] = v as u8;
        last[p.op[v].b as usize] = v as u8;
        v += 1;
    }
    let (mut free, mut nfree, mut slots) = ([0u8; MAX_VALUES], 0, 0);
    v = 0;
    while v < p.values {
        let s = if nfree > 0 {
            nfree -= 1;
            free[nfree]
        } else {
            slots += 1;
            (slots - 1) as u8
        };
        p.slot[v] = s;
        if v > 0 {
            let (a, b) = (p.op[v].a as usize, p.op[v].b as usize);
            if last[a] == v as u8 {
                free[nfree] = p.slot[a];
                nfree += 1;
            }
            if b != a && last[b] == v as u8 {
                free[nfree] = p.slot[b];
                nfree += 1;
            }
        }
        if last[v] == NONE {
            free[nfree] = s;
            nfree += 1;
        }
        v += 1;
    }
    p.slots = slots;
    p
}
