use crate::{
    add_buf, add_prim, cmp_buf, combine_u64, dec_buf, shl_buf, shr_buf, sub_buf, sub_prim,
    utils::div::{knuth_est, mul_u64_asm},
};

// Algorithm SqrtRemBaseβ(a₀, a₁, ..., aₙ₋₁)
//
// 1. If N = 0:
//        return (0, 0)
//
// 2. m ← ceil(n / 2)
//
// 3. Define aⱼ = 0 for j ≥ n
//
// 4. P ← a₂ₘ₋₁ β + a₂ₘ₋₂
//
// 5. S ← floor(sqrt(P))
// 6. R ← P - S²
//
// 7. for i ← m-2 down to 0:
//
//        h ← a₂ᵢ₊₁
//        l ← a₂ᵢ
//
//        // Estimate the next root limb.
//        A ← Rβ + h
//
//        q ← min(β - 1, floor(A / (2S)))
//
//        // Compute the exact residual for this candidate.
//        U ← A - 2Sq
//        E ← Uβ + l - q²
//
//        // Correct an overestimate.
//        while E < 0:
//            E ← E + 2(Sβ + q) - 1
//            q ← q - 1
//
//        // Accept the next root limb.
//        S ← Sβ + q
//        R ← E
//
// 8. return (S, R)

fn correct_sqrt(x: &mut [u64], s: &[u64], q_sqr: &[u64]) -> bool {
    let c = sub_buf(x, q_sqr);
    if c {
        add_buf(x, s);
        add_buf(x, s);
        dec_buf(x);
    }
    return c;
}

fn sqrt_4x2(x0: &mut u64, x1: &mut u64, x2: &mut u64, x3: &mut u64) -> (u64, u64) {
    let x_hi = combine_u64(*x2, *x3);
    let s_hi = x_hi.isqrt();
    let rem = x_hi - s_hi * s_hi;
    let c = rem / s_hi;
    let r0 = rem % s_hi;
    let y = (r0 << 64) | *x1 as u128;
    let t = y / s_hi;
    let v = y % s_hi;
    let (mut s_lo, d) = match c {
        0 => (t >> 1, t & 1),
        1 => ((1 << 63) | (t >> 1), t & 1),
        2 => (u64::MAX as u128, t + 2),
        _ => unreachable!(),
    };

    let u = v + (d as u128) * s_hi;
    let s_lo_sqr = s_lo * s_lo;
    let mut buf = [*x0, u as u64, (u >> 64) as u64];
    if correct_sqrt(
        &mut buf,
        &[s_lo as u64, s_hi as u64],
        &[s_lo_sqr as u64, (s_lo_sqr >> 64) as u64],
    ) {
        s_lo -= 1;
    }
    *x0 = buf[0];
    *x1 = buf[1];
    *x2 = buf[2];
    *x3 = 0;
    (s_lo as u64, s_hi as u64)
}

fn binom_sqrt_est(x: &mut [u64], s: &[u64], s0: u64, s1: u64) -> u64 {
    let mut c = 0;
    while cmp_buf(&x[1..], s).is_ge() {
        sub_buf(&mut x[1..], s);
        c += 1
    }
    let t = {
        let (win, of) = x.split_at_mut(s.len());
        knuth_est(win, &mut of[0], s, s1, s0)
    };
    let (q, d) = match c {
        0 => (t >> 1, t & 1),
        1 => ((1 << 63) | (t >> 1), t & 1),
        2 => (u64::MAX, t + 2),
        _ => unreachable!(),
    };
    for _ in 0..d {
        add_buf(x, s);
    }
    return q;
}

fn binom_sqrt(x: &mut [u64], s: &mut [u64]) {
    if s.is_empty() {
        return;
    }
    // assume correct size bounds
    debug_assert!(x.len() >= s.len());
    debug_assert!(2 * s.len() >= x.len());
    // assume normalized x
    debug_assert!(x.last().copied().unwrap() >= (1 << 62));

    let x_len = x.len();
    let s_len = s.len();
    s.fill(0);

    // x.len == 1 or 2
    if s_len == 1 {
        let term = if x_len == 1 {
            x[0] as u128
        } else {
            combine_u64(x[x_len - 2], x[x_len - 1])
        };
        s[s_len - 1] = term.isqrt() as u64;
        return;
    }

    let (mut x0, mut x1, mut x2, mut x3) = (
        x.get(x_len - 4).copied().unwrap_or(0),
        x.get(x_len - 3).copied().unwrap_or(0),
        x[x_len - 2],
        x[x_len - 1],
    );
    let (s0, s1) = sqrt_4x2(&mut x0, &mut x1, &mut x2, &mut x3);
    s[s_len - 2] = s0;
    s[s_len - 1] = s1;
    if s_len == 2 {
        return;
    }

    if let Some(e) = x.get_mut(x_len - 4) {
        *e = x0
    }
    if let Some(e) = x.get_mut(x_len - 3) {
        *e = x1
    }
    x[x_len - 2] = x2;
    x[x_len - 1] = x3;

    let mut i = x_len - 4;
    let mut j = s_len - 2;
    while j > 1 {
        while i >= 2 && j > 1 {
            i -= 2;
            j -= 1;

            s[j] = binom_sqrt_est(&mut x[i + 1..], &s[j + 1..], s0, s1);
            let s_sqr = unsafe { mul_u64_asm(s[j], s[j]) };
            if correct_sqrt(&mut x[i..], &s[j..], &[s_sqr.1, s_sqr.0]) {
                s[j] -= 1;
            }
        }
        if j > 1 {
            let cnt = x.iter().rev().take_while(|&&e| e == 0).count();
            x.copy_within(..x_len - cnt, cnt);
            x[..cnt].fill(0);
            i += cnt;
        }
    }

    i -= 1;
    j -= 1;

    s[j] = binom_sqrt_est(&mut x[i..], &s[j + 1..], s0, s1);
    let q_sqr = unsafe { mul_u64_asm(s[j], s[j]) };
    let corr = if i == 0 {
        cmp_buf(&x[i..], &[q_sqr.0 + u64::from(q_sqr.1 != 0)]).is_lt()
    } else {
        cmp_buf(&x[i - 1..], &[q_sqr.0, q_sqr.1]).is_lt()
    };
    if corr {
        s[j] -= 1;
    }
}
