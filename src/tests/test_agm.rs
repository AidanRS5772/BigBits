use super::{mul_ref, rand_nonzero_vec, rand_vec};
use crate::utils::consts::consts::{ln2_dyn, pi_dyn};
use crate::utils::sqrt::sqrt_only_dyn;
use crate::utils::trans::{agm_dyn, agm_dyn_at, agm_static, agm_static_at, theta_dyn, AGM_TAIL};
use crate::utils::utils::{add_buf, buf_len, cmp_buf, inc_buf, shr_buf, sub_buf};
use crate::utils::CapacityError;
use crate::utils::DYN_AGM_TAIL_TERMS;
use std::cmp::Ordering;
use std::f64::consts::LN_2;

fn bits(x: &[u64]) -> u64 {
    let n = buf_len(x);
    if n == 0 {
        0
    } else {
        64 * n as u64 - x[n - 1].leading_zeros() as u64
    }
}

// The n-limb window of v from limb lo, zero past its end.
fn window(v: &[u64], lo: usize, n: usize) -> Vec<u64> {
    (0..n).map(|i| v.get(lo + i).copied().unwrap_or(0)).collect()
}

// x + 2^e, x long enough.
fn add_pow2(x: &mut [u64], e: u64) {
    let carry = add_buf(&mut x[(e / 64) as usize..], &[1 << (e % 64)]);
    assert!(!carry);
}

// x >> n bits.
fn shr_bits(x: &[u64], n: usize) -> Vec<u64> {
    let mut v = window(x, n / 64, x.len());
    shr_buf(&mut v, (n % 64) as u8);
    v
}

// ⌊√x⌋ through sqrt_only_dyn, an odd length by its virtual low limb (a root 32 bits too high),
// checked by schoolbook: s² <= x < (s + 1)².
fn isqrt(x: &[u64]) -> Vec<u64> {
    let t = buf_len(x);
    let s = match t {
        0 => vec![0],
        1 => vec![x[0].isqrt()],
        _ => {
            let mut s = vec![0; t.div_ceil(2)];
            sqrt_only_dyn(&mut x[..t].to_vec(), &mut s);
            if t % 2 == 1 {
                shr_buf(&mut s, 32);
            }
            s
        }
    };
    let n = buf_len(&s);
    assert_ne!(cmp_buf(&mul_ref(&s[..n], &s[..n]), x), Ordering::Greater);
    let mut up = s.clone();
    up.push(0);
    inc_buf(&mut up);
    let n = buf_len(&up);
    assert_eq!(cmp_buf(&mul_ref(&up[..n], &up[..n]), x), Ordering::Greater);
    s
}

// log2 x for x > 0, from its top 64 bits.
fn log2(x: &[u64]) -> f64 {
    let lo = bits(x).saturating_sub(64);
    let top = shr_bits(x, lo as usize)[0];
    (top as f64).log2() + lo as f64
}

// log2(2^x + 2^y).
fn log2_sum(x: f64, y: f64) -> f64 {
    let (hi, lo) = (x.max(y), x.min(y));
    hi + (lo - hi).exp2().ln_1p() / LN_2
}

// The AGM's limit from exact floor means G limbs past an l-limb frame, run until a = b there and
// floored back to the frame, and the most a run in that frame may differ from it, for nonzero a
// and b. Each computed pair (a_k, b_k), k >= 1, is up to half a unit off in a (a floor) and a unit
// in b (a root), which moves the limit by α_k / 2 + β_k for (α_k, β_k) = ∂M / ∂(a_k, b_k). These
// come back from (1/2, 1/2) at the last pair through each step's Jacobian, in log2 so that a / b
// far past f64's range is fine: α_k = (α + β √(b_k / a_k)) / 2 and β_k = (α + β √(a_k / b_k)) / 2
// from k + 1's. A run stops earlier, so its pairs are a prefix of these. The bound is twice their
// sum plus the last mean's half unit, for the f64 and second-order slack, plus 2 for the oracle's
// own floor and error and TAIL_UNITS for the tail.
fn agm_oracle(a: &[u64], b: &[u64], l: usize) -> (Vec<u64>, Vec<u64>) {
    const G: usize = 2;
    let w = l + G + 1;
    let up = |x: &[u64]| {
        let mut v = vec![0; w];
        v[G..G + x.len()].copy_from_slice(x);
        v
    };
    let (mut a, mut b) = (up(a), up(b));
    // log2(b_k / a_k) for k = 1..n.
    let mut ratios = Vec::new();
    for k in 0.. {
        if k > 0 {
            ratios.push(log2(&b) - log2(&a));
        }
        if a == b {
            break;
        }
        assert!(k < 200, "the oracle converges");
        let g = isqrt(&mul_ref(&a[..buf_len(&a)], &b[..buf_len(&b)]));
        add_buf(&mut a, &b);
        shr_buf(&mut a, 1);
        b = window(&g, 0, w);
    }
    assert_eq!(buf_len(&a[G + l..]), 0);

    let (mut la, mut lb, mut total) = (-1.0, -1.0, -1.0_f64);
    // The last pair starts from (1/2, 1/2).
    for (i, &r) in ratios.iter().enumerate().rev() {
        if i + 1 < ratios.len() {
            (la, lb) = (
                log2_sum(la, lb + r / 2.0) - 1.0,
                log2_sum(la, lb - r / 2.0) - 1.0,
            );
        }
        total = log2_sum(total, log2_sum(la - 1.0, lb));
    }
    let mut tol = vec![0; w + 2];
    add_pow2(&mut tol, (total + 1.0).ceil() as u64);
    add_buf(&mut tol, &[2 + TAIL_UNITS]);
    (a[G..G + l].to_vec(), tol)
}

// The tail's own error (agm_tail: within 2 units of the limit of the pair it starts from).
const TAIL_UNITS: u64 = 2;

// The caps the tests run: every table length's neighborhood of the j = 0..4 boundaries.
const CAPS: [usize; 10] = [0, 1, 2, 3, 5, 7, 10, 15, 19, 40];

type AgmFn<'a> = &'a dyn Fn(&[u64], &[u64], usize, &mut [u64]) -> (usize, usize);

fn agm_static_fn<const N: usize>(
    a: &[u64],
    b: &[u64],
    cap: usize,
    out: &mut [u64],
) -> (usize, usize) {
    agm_static_at::<N>(a, b, cap, out).unwrap()
}

// agm into l limbs within the oracle's bound at every cap, with no more tail terms than the cap
// and no more roots as the cap grows. Returns each cap's (roots, terms).
fn check(agm: AgmFn, a: &[u64], b: &[u64], l: usize) -> Vec<(usize, usize)> {
    let (want, tol) = agm_oracle(a, b, l);
    let mut seen = Vec::new();
    for cap in CAPS {
        let mut got = vec![u64::MAX; l];
        let (roots, terms) = agm(a, b, cap, &mut got);
        let what = format!("l={l} cap={cap} a.len()={} b.len()={}", a.len(), b.len());
        assert!(terms <= cap.min(AGM_TAIL.len()), "{what}: {terms} terms");
        if let Some(&(r, _)) = seen.last() {
            assert!(roots <= r, "{what}: {roots} roots after {r}");
        }
        seen.push((roots, terms));
        let (mut diff, lo) = match cmp_buf(&got, &want) {
            Ordering::Less => (want.clone(), &got),
            _ => (got.clone(), &want),
        };
        sub_buf(&mut diff, lo);
        assert_ne!(
            cmp_buf(&diff, &tol),
            Ordering::Greater,
            "{what}: off by {:?}, bound {:?}",
            &diff[..buf_len(&diff)],
            &tol[..buf_len(&tol)]
        );
    }
    seen
}

// Across the approximate root's binomial/Zimmermann switch (11-limb roots) and the square
// shapes' School/Karatsuba (about 55 limbs) and Karatsuba/FFT (about 230) switches.
#[test]
fn test_agm_random_against_oracle() {
    let mut seed = 1;
    for l in [1, 2, 3, 4, 5, 10, 11, 12, 13, 40, 56, 100, 240, 300] {
        for _ in 0..2 {
            let (a, b) = (rand_nonzero_vec(l, seed), rand_nonzero_vec(l, seed + 1));
            seed += 2;
            let seen = check(&agm_dyn_at, &a, &b, l);
            // The default is one of the caps.
            let mut got = vec![0; l];
            agm_dyn(&a, &b, &mut got);
            let mut at = vec![0; l];
            agm_dyn_at(&a, &b, DYN_AGM_TAIL_TERMS, &mut at);
            assert_eq!(got, at, "l={l}");
            // Past a few limbs the tail replaces roots.
            if l >= 40 {
                assert!(seen[0].0 > seen[7].0, "l={l}: {seen:?}");
            }
        }
    }
}

// b far below a, with every count of zero top limbs: odd and even product lengths, and errors in
// b that cost about a / b.
#[test]
fn test_agm_small_b() {
    let mut seed = 1000;
    for l in [2, 3, 4, 5, 12, 13, 40] {
        for k in 1..l {
            let a = rand_nonzero_vec(l, seed);
            let mut b = rand_nonzero_vec(l - k, seed + 1);
            seed += 2;
            check(&agm_dyn_at, &a, &b, l);
            b.resize(l, 0);
            check(&agm_dyn_at, &b, &a, l);
        }
    }
    check(&agm_dyn_at, &rand_nonzero_vec(20, 7), &[1], 20);
}

// a = B^(l-1) makes the first product b B^(l-1), so b's length sets its parity (odd ones take
// the padded root and its 32-bit shift) and b's top limb its top limb, from 63 leading zeros
// to none.
#[test]
fn test_agm_product_parity() {
    let mut seed = 2000;
    for l in [2, 3, 4, 5, 12, 13, 24, 25] {
        let mut a = vec![0; l];
        a[l - 1] = 1;
        for lb in 1..=l {
            for top in [1, 2, 3, 4, 7, 1 << 32, 1 << 63, u64::MAX] {
                let mut b = rand_vec(lb, seed);
                seed += 1;
                b[lb - 1] = top;
                check(&agm_dyn_at, &a, &b, l);
            }
        }
    }
}

// Products of one limb, which the root takes as x B like any odd length, in one-limb and wider
// frames.
#[test]
fn test_agm_one_limb_product() {
    for (a, b) in [(3, 5), (1, 1), (1, 2), (7, 9), (1 << 20, 1 << 30), (u32::MAX as u64, 3)] {
        for l in [1, 2, 3] {
            check(&agm_dyn_at, &[a], &[b], l);
            check(&agm_static_fn::<6>, &[a], &[b], l);
        }
    }
}

#[test]
fn test_agm_exact_cases() {
    let mut seed = 3000;
    for l in [1, 2, 5, 13, 40] {
        let (a, b) = (rand_nonzero_vec(l, seed), rand_nonzero_vec(l, seed + 1));
        seed += 2;
        let zero = vec![0; l];
        for cap in CAPS {
            let mut out = vec![7; l];
            // AGM(x, 0) = 0 with no steps.
            assert_eq!(agm_dyn_at(&a, &zero, cap, &mut out), (0, 0));
            assert_eq!(out, zero, "b = 0, l={l}");
            out.fill(7);
            agm_dyn_at(&zero, &b, cap, &mut out);
            assert_eq!(out, zero, "a = 0, l={l}");
            // AGM(a, a) = a, and a and a + 1 give a with no steps either.
            assert_eq!(agm_dyn_at(&a, &a, cap, &mut out), (0, 0));
            assert_eq!(out, a, "a = b, l={l}");
            let mut a1 = a.clone();
            a1.push(0);
            inc_buf(&mut a1);
            let a1 = &a1[..buf_len(&a1).max(l)];
            if a1.len() == l {
                agm_dyn_at(&a, a1, cap, &mut out);
                assert_eq!(out, a, "b = a + 1, l={l}");
            }
            agm_dyn_at(&a, &b, cap, &mut out);
            let mut swapped = vec![0; l];
            agm_dyn_at(&b, &a, cap, &mut swapped);
            assert_eq!(out, swapped, "symmetry, l={l} cap={cap}");
            // Shorter inputs are zero-extended.
            let mut wide = vec![0; l + 3];
            agm_dyn_at(&a, &b, cap, &mut wide);
            let mut wide_ref = vec![0; l + 3];
            agm_dyn_at(&[a.clone(), vec![0; 3]].concat(), &b, cap, &mut wide_ref);
            assert_eq!(wide, wide_ref, "wider out, l={l} cap={cap}");
        }
        check(&agm_dyn_at, &vec![u64::MAX; l], &vec![u64::MAX - 1; l], l);
        check(&agm_dyn_at, &vec![u64::MAX; l], &a, l);
    }
}

// Pairs that start inside the tail, at every term count: A random with its top bit set and
// B = A - 2 D for a D of each width, so the gap t = bits(m) - bits(D) - 1 sweeps the range where
// the series needs 0 to 19 terms, including t at and around multiples of 64 (ε's zero limbs).
#[test]
fn test_agm_tail_every_term_count() {
    let mut seed = 6000;
    for l in [1, 2, 3, 8, 40] {
        let bm = 64 * l as u64;
        let mut counts = vec![false; AGM_TAIL.len() + 1];
        let mut a = rand_nonzero_vec(l, seed);
        a[l - 1] |= 1 << 63;
        for bd in (1..bm - 2).rev() {
            let t = bm - bd - 1;
            if t > 2 * (bm / 128) + 66 && t % 64 > 2 && t % 64 < 62 && bd % 7 != 0 {
                continue;
            }
            let mut dv = rand_vec(l, seed + bd);
            seed += 1;
            let (w, v) = ((bd / 64) as usize, bd % 64);
            dv[w..].fill(0);
            dv[w] = if v == 0 { 0 } else { dv[w] & ((1 << v) - 1) };
            dv[(bd - 1) as usize / 64] |= 1 << ((bd - 1) % 64);
            let mut b = a.clone();
            let mut d2 = dv.clone();
            d2.push(0);
            shl_one(&mut d2);
            if buf_len(&d2) > l || cmp_buf(&d2[..l], &b) != Ordering::Less {
                continue;
            }
            sub_buf(&mut b, &d2[..l]);
            for (x, y) in [(&a, &b), (&b, &a)] {
                for (roots, terms) in check(&agm_dyn_at, x, y, l) {
                    if roots == 0 {
                        counts[terms] = true;
                    }
                }
            }
        }
        // Narrower frames step past some counts: I ~ bits(m) / 2t moves by more than 1 a bit.
        if l >= 40 {
            assert!(counts.iter().all(|&c| c), "l={l}: {counts:?}");
        }
    }
}

// x << 1 in place, x long enough.
fn shl_one(x: &mut [u64]) {
    let mut carry = 0;
    for l in x.iter_mut() {
        let c = *l >> 63;
        *l = *l << 1 | carry;
        carry = c;
    }
    assert_eq!(carry, 0);
}

// AGM_TAIL against the inverse of K's series Σ k_i u^i, k_i = C(2i, i)² / 2^(4i), from
// g_n = k_n - Σ_(0<i<n) k_i g_(n-i), exactly, at 2^(4 (len + 1)) (every product's denominator
// divides 2^(4n)).
#[test]
fn test_agm_tail_coefficients() {
    let s = 4 * (AGM_TAIL.len() as u64 + 1);
    let binom = |n: u64, k: u64| (0..k).fold(1u128, |c, i| c * (n - i) as u128 / (i + 1) as u128);
    let scaled = |x: u128, e: u64| {
        // x 2^(s - e), as limbs.
        let mut v = vec![x as u64, (x >> 64) as u64, 0, 0];
        for _ in 0..s - e {
            shl_one(&mut v);
        }
        v
    };
    let mut g: Vec<Vec<u64>> = vec![Vec::new()];
    for n in 1..=AGM_TAIL.len() as u64 {
        let c = binom(2 * n, n);
        let mut acc = mul_ref(&scaled(c, 4 * n), &[c as u64, (c >> 64) as u64]);
        acc.resize(8, 0);
        for i in 1..n {
            let ci = binom(2 * i, i);
            let mut p = mul_ref(&mul_ref(&g[(n - i) as usize], &[ci as u64]), &[ci as u64]);
            let whole = shr_bits(&p, 4 * i as usize);
            let mut back = whole.clone();
            for _ in 0..4 * i {
                back.push(0);
                shl_one(&mut back);
            }
            assert_eq!(
                back[..buf_len(&back)],
                p[..buf_len(&p)],
                "k_{i} g_{} is whole",
                n - i
            );
            p = whole;
            p.resize(8, 0);
            assert!(!sub_buf(&mut acc, &p[..8]));
        }
        let (num, e) = AGM_TAIL[n as usize - 1];
        let want = scaled(num as u128, e as u64);
        assert_eq!(acc[..buf_len(&acc)], want[..buf_len(&want)], "g_{n}");
        acc.truncate(buf_len(&acc));
        g.push(acc);
        if n > 1 {
            let (p, pe) = AGM_TAIL[n as usize - 2];
            assert!(
                (num as u128) << pe < (p as u128) << e,
                "g_{n} < g_{}",
                n - 1
            );
        }
    }
}

#[test]
#[should_panic]
fn test_agm_input_longer_than_out_panics() {
    agm_dyn(&[1, 2, 3], &[1], &mut [0; 2]);
}

fn check_static<const N: usize>(seed: &mut u64) {
    let f: AgmFn = &agm_static_fn::<N>;
    for l in [1, 2, N / 4, N / 2, N - 3, N - 2] {
        if l == 0 || l + 2 > N {
            continue;
        }
        let (a, b) = (rand_nonzero_vec(l, *seed), rand_nonzero_vec(l, *seed + 1));
        let small = rand_nonzero_vec(l.div_ceil(3), *seed + 2);
        *seed += 3;
        check(f, &a, &b, l);
        check(f, &a, &small, l);
        // The dynamic run agrees within twice the bound.
        let (_, tol) = agm_oracle(&a, &b, l);
        for cap in CAPS {
            let (mut s, mut d) = (vec![0; l], vec![0; l]);
            agm_static_at::<N>(&a, &b, cap, &mut s).unwrap();
            agm_dyn_at(&a, &b, cap, &mut d);
            let (mut diff, lo) = match cmp_buf(&s, &d) {
                Ordering::Less => (d.clone(), &s),
                _ => (s.clone(), &d),
            };
            sub_buf(&mut diff, lo);
            let mut tol2 = tol.clone();
            add_buf(&mut tol2, &tol);
            assert_ne!(
                cmp_buf(&diff, &tol2),
                Ordering::Greater,
                "N={N} l={l} cap={cap}"
            );
        }
    }
}

#[test]
fn test_agm_static_against_oracle() {
    let mut seed = 4000;
    check_static::<3>(&mut seed);
    check_static::<8>(&mut seed);
    check_static::<26>(&mut seed);
    check_static::<128>(&mut seed);
    check_static::<600>(&mut seed);
}

#[test]
fn test_agm_static_capacity() {
    let (a, b) = (rand_nonzero_vec(4, 5000), rand_nonzero_vec(4, 5001));
    let mut out = [7; 4];
    assert_eq!(
        agm_static::<5>(&a, &b, &mut out),
        Err(CapacityError { min_n: 6 })
    );
    assert_eq!(out, [7; 4]);
    assert_eq!(agm_static::<6>(&a, &b, &mut out), Ok(()));
}

// ln(1/q) = π / (4 AGM(θ2², θ3²)) at q = 2^-s. theta_dyn's outputs are squared into one frame
// of f' fraction limbs, with θ2²'s 2s leading zero bits and a guard limb past p = 64 f bits. The
// AGM's limit M must satisfy 4 s M ln 2 = π to within 2^(8 - p) relative.
fn check_ln_identity(agm: &dyn Fn(&[u64], &[u64], &mut [u64]), f: usize, s: u64) {
    let fp = f + (2 * s).div_ceil(64) as usize + 1;
    let l = fp + 1;
    // θ2 2^(64 (w2 + z)) at about the bits θ2² keeps in the frame, two limbs over.
    let z = ((s - 3) / 64) as usize;
    let w2 = fp - 2 * z + 2;
    let (mut th2, mut th3) = (vec![0; w2], vec![0; l]);
    assert_eq!(theta_dyn::<16>(&[1], s, &mut th2, &mut th3), z);
    let a = window(&mul_ref(&th3, &th3), fp, l);
    let b = window(&mul_ref(&th2, &th2), 2 * (w2 + z) - fp, l);

    let mut m = vec![0; l];
    agm(&b, &a, &mut m);

    let (mut ln2, mut pi) = (vec![0; fp], vec![0; fp + 1]);
    ln2_dyn(&mut ln2);
    pi_dyn(&mut pi);
    let mut lhs = mul_ref(&mul_ref(&m[..buf_len(&m)], &ln2), &[4 * s]);
    let mut rhs = vec![0; fp];
    rhs.extend_from_slice(&pi);
    let w = lhs.len().max(rhs.len());
    lhs.resize(w, 0);
    rhs.resize(w, 0);
    let (mut diff, lo) = match cmp_buf(&lhs, &rhs) {
        Ordering::Less => (rhs.clone(), &lhs),
        _ => (lhs.clone(), &rhs),
    };
    sub_buf(&mut diff, lo);
    // π B^(2 f') < 2^(128 f' + 2).
    let bound = 128 * fp as u64 + 2 + 8 - 64 * f as u64;
    assert!(
        bits(&diff) <= bound,
        "f={f} s={s}: error 2^{} past 2^{bound}",
        bits(&diff)
    );
}

#[test]
fn test_agm_theta_ln_identity() {
    for f in [1, 2, 8, 32] {
        for s in [3, 4, 64, 67, 200, 1000] {
            check_ln_identity(&agm_dyn, f, s);
        }
    }
    let st = |a: &[u64], b: &[u64], o: &mut [u64]| agm_static::<200>(a, b, o).unwrap();
    check_ln_identity(&st, 32, 1000);
    check_ln_identity(&st, 8, 67);
}
