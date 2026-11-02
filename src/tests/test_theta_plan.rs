use crate::utils::theta_plan::{ThetaPlan, ThetaPlans, MAX_VALUES, THETA_MAX_TERMS};

// Every value's exponent is its operands' sum, operands come earlier, and the terms are exactly
// q^(n²) for n = 1..=2T + 1, each once.
fn check_valid(p: &ThetaPlan, t: usize, what: &str) {
    let k = 2 * t + 1;
    assert!(p.values >= 1 && p.values <= MAX_VALUES, "{what}");
    assert_eq!((p.exp[0], p.op[0].term), (1, 1), "{what}: value 0 is q");
    let mut seen = vec![false; k + 1];
    seen[1] = true;
    for v in 1..p.values {
        let o = p.op[v];
        let (a, b) = (o.a as usize, o.b as usize);
        assert!(a < v && b < v, "{what}: value {v} reads a later value");
        assert_eq!(p.exp[v], p.exp[a] + p.exp[b], "{what}: value {v}");
        let e = p.exp[v] as usize;
        let n = (1..=k).find(|&n| n * n == e).unwrap_or(0);
        assert_eq!(o.term as usize, n, "{what}: value {v}'s term");
        if n != 0 {
            assert!(!seen[n], "{what}: term {n} twice");
            seen[n] = true;
        }
    }
    assert!(seen.iter().skip(1).all(|&s| s), "{what}: a term is missing");
}

// Running the ops in order, no slot is written while its value still has a reader, and slots is
// the most values alive at once (an operand is alive while the op reading it last runs).
fn check_slots(p: &ThetaPlan, what: &str) {
    let mut last = vec![0; p.values];
    for v in 1..p.values {
        last[p.op[v].a as usize] = v;
        last[p.op[v].b as usize] = v;
    }
    let mut owner = vec![None; p.slots];
    let mut peak = 0;
    for v in 0..p.values {
        let s = p.slot[v] as usize;
        assert!(s < p.slots, "{what}");
        if let Some(u) = owner[s] {
            assert!(
                last[u] < v,
                "{what}: value {v} overwrites value {u}, read at {}",
                last[u]
            );
        }
        owner[s] = Some(v);
        peak = peak.max(1 + (0..v).filter(|&u| last[u] >= v).count());
    }
    assert_eq!(peak, p.slots, "{what}");
}

// Width-weighted work (squares 2, products 3, each times the window of the first term it feeds)
// and the dependency path's finish time with the same unit costs.
fn costs(p: &ThetaPlan, t: usize) -> (u64, u64) {
    let c = 4 * (t as u64 + 1).pow(2);
    let weight = |n: u64| c - (n * n - (n & 1));
    // The first term each value feeds is the smallest one reachable from it.
    let mut first = vec![u64::MAX; p.values];
    for v in (0..p.values).rev() {
        if p.op[v].term != 0 {
            first[v] = first[v].min(p.op[v].term as u64);
        }
        if v > 0 {
            for i in [p.op[v].a, p.op[v].b] {
                first[i as usize] = first[i as usize].min(first[v]);
            }
        }
    }
    let (mut work, mut fin) = (0, vec![0; p.values]);
    for v in 1..p.values {
        let o = p.op[v];
        let d = if o.a == o.b { 2 } else { 3 };
        work += d * weight(first[v]);
        fin[v] = fin[o.a as usize].max(fin[o.b as usize]) + d;
    }
    (work, fin.into_iter().max().unwrap_or(0))
}

// The current recurrence's work: q², then B_n = B_(n-1) q² and A_n = A_(n-1) B_n per n.
fn recurrence_work(t: usize) -> u64 {
    let c = 4 * (t as u64 + 1).pow(2);
    let weight = |n: u64| c - (n * n - (n & 1));
    if t == 0 {
        return 0;
    }
    2 * weight(2) + (2..=2 * t as u64 + 1).map(|n| 6 * weight(n)).sum::<u64>()
}

fn check<const T: usize>() {
    let (s, p) = (ThetaPlans::<T>::SERIAL, ThetaPlans::<T>::PARALLEL);
    for (plan, name) in [(&s, "serial"), (&p, "parallel")] {
        let what = format!("T={T} {name}");
        check_valid(plan, T, &what);
        check_slots(plan, &what);
    }
    let ((sw, ss), (pw, ps)) = (costs(&s, T), costs(&p, T));
    assert!(
        sw <= recurrence_work(T),
        "T={T}: serial work {sw} past the recurrence's"
    );
    assert!(sw <= pw, "T={T}: serial work {sw} past parallel's {pw}");
    assert!(ps <= ss, "T={T}: parallel span {ps} past serial's {ss}");
    assert!(p.width >= 1 && p.width <= p.values, "T={T}");
}

macro_rules! check_all {
    ($($t:literal)*) => { $(check::<$t>();)* };
}

#[test]
fn test_theta_plans_valid() {
    check_all!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    assert_eq!(THETA_MAX_TERMS, 16);
}

fn exps(p: &ThetaPlan) -> Vec<u16> {
    p.exp[1..p.values].to_vec()
}

// From the scratch reference (integer costs, first minimum kept).
#[test]
fn test_theta_plans_t3() {
    let (s, p) = (ThetaPlans::<3>::SERIAL, ThetaPlans::<3>::PARALLEL);
    assert_eq!(exps(&s), [2, 4, 8, 9, 16, 25, 18, 36, 13, 49]);
    assert_eq!(exps(&p), [2, 4, 8, 9, 16, 17, 32, 25, 36, 49]);
    // 18 = 9 + 9 and 36 = 18 + 18 are squares, 13 = 4 + 9 a product.
    assert_eq!(
        s.op[7],
        crate::utils::theta_plan::ThetaOp {
            a: 4,
            b: 4,
            term: 0
        }
    );
    assert_eq!(s.op[8].a, s.op[8].b);
    assert_ne!(s.op[9].a, s.op[9].b);
    assert_eq!((s.slots, p.slots), (4, 6));
    assert_eq!(costs(&s, 3).0 * 1000 / recurrence_work(3), 587);
}
