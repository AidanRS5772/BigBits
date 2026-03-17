use std::arch::asm;

#[inline(always)]
pub(crate) fn trim_lz(vec: &mut Vec<u64>) {
    if let Some(idx) = vec.iter().rposition(|&x| x != 0) {
        vec.truncate(idx + 1);
    } else {
        vec.clear();
    }
}

#[inline(always)]
pub(crate) fn buf_len(val: &[u64]) -> usize {
    return if let Some(idx) = val.iter().rposition(|&x| x != 0) {
        idx + 1
    } else {
        0
    };
}

#[inline(always)]
pub(crate) fn scmp(s: bool, ord: std::cmp::Ordering) -> std::cmp::Ordering {
    return if s { ord.reverse() } else { ord };
}

#[inline(always)]
pub(crate) fn push_prim2(buf: &mut Vec<u64>, mut prim: u128) {
    if prim > 0 {
        buf.push(prim as u64);
        prim >>= 64;
        if prim > 0 {
            buf.push(prim as u64);
        }
    }
}

#[inline(always)]
fn combine_u64(x0: u64, x1: u64) -> u128 {
    x0 as u128 | (x1 as u128) << 64
}

pub(crate) fn eq_buf(lhs: &[u64], rhs: &[u64]) -> bool {
    match lhs.len().cmp(&rhs.len()) {
        Greater => {
            return lhs[rhs.len()..].iter().any(|&x| x != 0);
        }
        Less => {
            return rhs[lhs.len()..].iter().any(|&x| x != 0);
        }
        Equal => {}
    }

    for (l, r) in lhs.iter().zip(rhs) {
        if l != r {
            return false;
        }
    }
    return true;
}

pub(crate) fn cmp_buf(lhs: &[u64], rhs: &[u64]) -> std::cmp::Ordering {
    match lhs.len().cmp(&rhs.len()) {
        Greater => {
            if lhs[rhs.len()..].iter().any(|&x| x != 0) {
                return Greater;
            }
        }
        Less => {
            if rhs[lhs.len()..].iter().any(|&x| x != 0) {
                return Less;
            }
        }
        Equal => {}
    }

    for (l, r) in lhs.iter().zip(rhs).rev() {
        let cmp = l.cmp(r);
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }

    return std::cmp::Ordering::Equal;
}

#[inline]
pub(crate) fn signed_shl(val: u64, sh: i32) -> u64 {
    return if sh < 0 {
        val >> sh.unsigned_abs()
    } else {
        val << sh.unsigned_abs()
    };
}

#[inline]
pub(crate) fn signed_shr(val: u64, sh: i32) -> u64 {
    return if sh < 0 {
        val << sh.unsigned_abs()
    } else {
        val >> sh.unsigned_abs()
    };
}

#[inline]
pub(crate) fn lsb(val: u64, sh: i32) -> u64 {
    return if sh <= 0 {
        0
    } else {
        val & ((1 << sh.unsigned_abs()) - 1)
    };
}

// adds values with carry and propagates carry on ARM
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn add_with_carry_aarch64(l: &mut u64, s: u64, c: &mut u8) {
    asm!(
        "subs wzr, {c:w}, #1", // c -> cf
        "adcs {l}, {l}, {s}", // l+s+cf -> l , updates cf
        "cset {c:w}, cs", // cf -> c
        l = inout(reg) *l,
        s = in(reg) s,
        c = inout(reg) *c,
        options(nostack)
    );
}

// adds values with carry and propagates carry on x86
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn add_with_carry_x86_64(l: &mut u64, s: u64, c: &mut u8) {
    asm!(
        "add {c} , 0xFF", // c -> cc
        "adc {l}, {s}", // l+s+cf -> l , updates cf
        "setc {c}", // cf -> c
        c = inout(reg_byte) *c,
        l = inout(reg) *l,
        s = in(reg) s,
        options(nostack)
    );
}

//architecture wrapper
#[inline(always)]
unsafe fn add_with_carry(l: &mut u64, s: u64, c: &mut u8) {
    #[cfg(target_arch = "aarch64")]
    add_with_carry_aarch64(l, s, c);

    #[cfg(target_arch = "x86_64")]
    add_with_carry_x86_64(l, s, c);
}

pub(crate) fn acc(lhs: &mut [u64], rhs: &[u64], comp: u8) -> bool {
    let mask = if comp == 0 { 0 } else { u64::MAX };
    let cf = comp;
    unsafe {
        let mut c = comp;
        for (l, s) in lhs.iter_mut().zip(rhs) {
            add_with_carry(l, *s ^ mask, &mut c);
        }
        if c != cf {
            for l in &mut lhs[rhs.len()..] {
                add_with_carry(l, mask, &mut c);
                if c == cf {
                    break;
                }
            }
        }

        return c != cf;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn inc_propagate_aarch64(l: &mut u64, c: &mut u8) {
    asm!(
        "adds {l}, {l}, {c}", // l + c -> l, sets CF
        "cset {c:w}, cs",     // CF -> c
        l = inout(reg) * l,
        c = inout(reg) * c,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn inc_propagate_x86_64(l: &mut u64, c: &mut u8) {
    asm!(
        "add {c}, 0xFF", // c -> CF (0xFF + 1 overflows, 0xFF + 0 doesn't)
        "adc {l}, 0",    // l + 0 + CF -> l
        "setc {c}",      // CF -> c
        c = inout(reg_byte) * c,
        l = inout(reg) * l,
        options(nostack)
    );
}

#[inline(always)]
unsafe fn inc_propagate(l: &mut u64, c: &mut u8) {
    #[cfg(target_arch = "aarch64")]
    inc_propagate_aarch64(l, c);
    #[cfg(target_arch = "x86_64")]
    inc_propagate_x86_64(l, c);
}

pub(crate) fn inc(lhs: &mut [u64]) -> bool {
    unsafe {
        let mut c: u8 = 1;
        for l in lhs.iter_mut() {
            inc_propagate(l, &mut c);
            if c == 0 {
                return false;
            }
        }
        true
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dec_propagate_aarch64(l: &mut u64, b: &mut u8) {
    asm!(
        "subs {l}, {l}, {b}", // l - b -> l, sets C = NOT borrow
        "cset {b:w}, cc",     // cc (carry clear) = borrow occurred -> b
        l = inout(reg) * l,
        b = inout(reg) * b,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dec_propagate_x86_64(l: &mut u64, b: &mut u8) {
    asm!(
        "add {b}, 0xFF", // b -> CF
        "sbb {l}, 0",    // l - 0 - CF -> l
        "setc {b}",      // CF -> b
        b = inout(reg_byte) * b,
        l = inout(reg) * l,
        options(nostack)
    );
}

#[inline(always)]
unsafe fn dec_propagate(l: &mut u64, b: &mut u8) {
    #[cfg(target_arch = "aarch64")]
    dec_propagate_aarch64(l, b);
    #[cfg(target_arch = "x86_64")]
    dec_propagate_x86_64(l, b);
}

pub(crate) fn dec(lhs: &mut [u64]) -> bool {
    unsafe {
        let mut b: u8 = 1;
        for l in lhs.iter_mut() {
            dec_propagate(l, &mut b);
            if b == 0 {
                return false;
            }
        }
        true
    }
}
pub(crate) fn twos_comp(buf: &mut [u64]) {
    for l in buf {
        *l = !*l;
    }
    inc(buf);
}

pub(crate) fn mul_prim(buf: &mut [u64], prim: u64) -> u64 {
    let prim_u128 = prim as u128;
    let mut carry: u128 = 0;

    for e in buf {
        let val = (*e as u128) * prim_u128 + carry;
        *e = val as u64;
        carry = val >> 64;
    }

    return carry as u64;
}

pub(crate) fn mul_prim2(buf: &mut [u64], prim: u128) -> u128 {
    let mask = u64::MAX as u128;

    let p0 = prim & mask;
    let p1 = prim >> 64;

    let last = *buf.last().unwrap() as u128;

    let val_u128 = (buf[0] as u128) * p0;
    let mut carry = val_u128 >> 64;
    let mut val = val_u128 as u64;

    for i in 1..buf.len() {
        let mut term = carry;
        carry = 0;

        let val0 = (buf[i - 1] as u128) * p1;
        term += val0 & mask;
        carry += val0 >> 64;

        let val1 = (buf[i] as u128) * p0;
        term += val1 & mask;
        carry += val1 >> 64;

        carry += term >> 64;
        buf[i - 1] = val;
        val = term as u64;
    }
    buf[buf.len() - 1] = val;

    return last * p1 + carry;
}

pub(crate) fn mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let a_len = a.len() - 1;
    let b_len = b.len() - 1;

    let mask = u64::MAX as u128;
    let mut carry: u128 = 0;
    for n in 0..out.len() {
        let mut term: u128 = carry;
        carry = 0;
        let mi = n.saturating_sub(b_len);
        let mf = n.min(a_len);
        for m in mi..=mf {
            let val = (a[m] as u128) * (b[n - m] as u128);
            term += val & mask;
            carry += val >> 64;
        }
        carry += term >> 64;
        out[n] = term as u64;
    }

    return carry as u64;
}

fn chunking_karatsuba(long: &[u64], short: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    out.fill(0);
    let s = short.len();
    let mut overflow = 0;

    let n_chunks = long.len() / s;
    let len = if long.len() % s == 0 {
        n_chunks - 1
    } else {
        n_chunks
    };

    for i in 0..len {
        let (val, rest) = scratch.split_at_mut(2 * s);
        let of = i * s;
        karatsuba_alg(&long[of..of + s], short, val, rest);
        acc(&mut out[of..], val, 0) as u64;
    }

    let of = n_chunks * s;
    let chunk = &long[of..];
    let (val, rest) = scratch.split_at_mut(chunk.len() + s - 1);
    overflow += karatsuba_alg(short, chunk, val, rest);
    if acc(&mut out[of..], val, 0) {
        overflow += 1;
    }

    return overflow;
}

enum KDispatch {
    Prim,
    Prim2,
    Base,
    Chunking,
    Recurse,
}

const KARATSUBA_CUTOFF: usize = 32;

fn karatsuba_dispatch(long: usize, short: usize) -> KDispatch {
    return if short == 1 {
        KDispatch::Prim
    } else if short == 2 {
        KDispatch::Prim2
    } else if long <= KARATSUBA_CUTOFF {
        KDispatch::Base
    } else if short < (long + 1) / 2 {
        if short <= KARATSUBA_CUTOFF {
            KDispatch::Base
        } else {
            KDispatch::Chunking
        }
    } else {
        KDispatch::Recurse
    };
}

fn karatsuba_core(
    long: &[u64],
    short: &[u64],
    half_len: usize,   // (long + 1) / 2
    out: &mut [u64],   // >= long + short - 1
    cross: &mut [u64], // >= 2*half + 1
    scratch: &mut [u64],
) -> u64 {
    let (l0, l1) = long.split_at(half_len);
    let (s0, s1) = short.split_at(half_len);

    {
        let (l_sum, s_sum) = out[..2 * half_len + 2].split_at_mut(half_len + 1);
        let mut l_len = half_len;
        let mut s_len = half_len;

        l_sum[..l_len].copy_from_slice(l0);
        if acc(&mut l_sum[..l_len], l1, 0) {
            l_sum[l_len] = 1;
            l_len += 1;
        }

        s_sum[..s_len].copy_from_slice(s0);
        if acc(&mut s_sum[..s_len], s1, 0) {
            s_sum[s_len] = 1;
            s_len += 1;
        }

        karatsuba_alg(&l_sum[..l_len], &s_sum[..s_len], cross, scratch);
    }

    let (z0, z2) = out.split_at_mut(2 * half_len);
    karatsuba_alg(l0, s0, z0, scratch);
    let c = karatsuba_alg(l1, s1, z2, scratch);

    acc(cross, z0, 1);
    acc(cross, z2, 1);
    if c > 0 {
        acc(&mut cross[z2.len()..], &[c], 1);
    }

    return c + acc(&mut out[half_len..], &cross, 0) as u64;
}

fn karatsuba_alg(long: &[u64], short: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    match karatsuba_dispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            return mul_prim(out, short[0]);
        }
        KDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            out[out.len() - 1] = 0;
            return mul_prim2(out, short[0] as u128 | (short[0] as u128) << 64) as u64;
        }
        KDispatch::Base => {
            return mul_buf(long, short, out);
        }
        KDispatch::Chunking => {
            return chunking_karatsuba(long, short, out, scratch);
        }
        KDispatch::Recurse => {}
    }

    let half_len = (long.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
    cross.fill(0);

    karatsuba_core(long, short, half_len, out, cross, rest)
}

fn find_karatsuba_scratch_size(l: usize, s: usize) -> usize {
    if l <= KARATSUBA_CUTOFF || s <= 2 {
        return 0;
    }

    let init_half = (l + 1) / 2;

    let (mut n, mut total) = if s < init_half {
        if s <= KARATSUBA_CUTOFF {
            return 0;
        }
        (s, 2 * s)
    } else {
        (init_half + 1, 2 * init_half + 1)
    };

    while n > KARATSUBA_CUTOFF {
        let half = (n + 1) / 2;
        total += 2 * half + 1;
        n = half + 1;
    }

    total
}

pub(crate) fn mul_vec(a: &[u64], b: &[u64]) -> (Vec<u64>, u64) {
    let a_len = a.len();
    let b_len = b.len();

    let mut out = vec![0_u64; a.len() + b.len() - 1];
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let c = match karatsuba_dispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            mul_prim(&mut out, short[0])
        }
        KDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            mul_prim2(&mut out, combine_u64(short[0], short[1])) as u64
        }
        KDispatch::Base => mul_buf(long, short, &mut out),
        KDispatch::Chunking => {
            let mut scratch = vec![0_u64; find_karatsuba_scratch_size(long.len(), short.len())];
            chunking_karatsuba(long, short, &mut out, &mut scratch)
        }
        KDispatch::Recurse => {
            let mut cross_scratch =
                vec![0_u64; find_karatsuba_scratch_size(long.len(), short.len())];
            let half = (long.len() + 1) / 2;
            let (cross, scratch) = cross_scratch.split_at_mut(2 * half + 1);
            karatsuba_core(long, short, half, &mut out, cross, scratch)
        }
    };

    return (out, c);
}

pub(crate) fn mul_arr<const N: usize>(a: &[u64], b: &[u64]) -> Result<([u64; N], u64), ()> {
    let a_len = buf_len(a);
    let b_len = buf_len(b);
    if a_len + b_len - 1 > N {
        return Err(());
    }

    let mut out = [0_u64; N];
    let (long, short) = if a_len > b_len { (a, b) } else { (b, a) };
    let c = match karatsuba_dispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            mul_prim(&mut out, short[0])
        }
        KDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            mul_prim2(&mut out, combine_u64(short[0], short[1])) as u64
        }
        KDispatch::Base => mul_buf(long, short, &mut out),
        KDispatch::Chunking => {
            let mut scratch = [0; N];
            chunking_karatsuba(long, short, &mut out, &mut scratch)
        }
        KDispatch::Recurse => {
            let mut scratch = [0; N];
            let scratch_sz = find_karatsuba_scratch_size(long.len(), short.len());
            let half_len = (long.len() + 1) / 2;
            if scratch_sz > N {
                let mut cross = [0; N];
                karatsuba_core(long, short, half_len, &mut out, &mut cross, &mut scratch)
            } else {
                let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
                karatsuba_core(long, short, half_len, &mut out, cross, rest)
            }
        }
    };
    return Ok((out, c));
}

pub(crate) fn sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 {
    let len = buf.len() - 1;
    let mask = u64::MAX as u128;

    let init_term = (buf[0] as u128) * (buf[0] as u128);
    out[0] = init_term as u64;
    let mut carry = init_term >> 64;
    for n in 1..out.len() {
        let mut term = carry;
        carry = 0;

        let mi = n.saturating_sub(len);
        let mf = ((n - 1) / 2).min(len);
        for m in mi..=mf {
            let val = (buf[m] as u128) * (buf[n - m] as u128);
            term += (val << 1) & mask;
            carry += val >> 63;
        }

        if n % 2 == 0 {
            let half_val = buf[n / 2] as u128;
            let val = half_val * half_val;
            term += val & mask;
            carry += val >> 64;
        }

        out[n] = term as u64;
        carry += term >> 64;
    }

    return carry as u64;
}

const KARATSUBA_SQR_CUTOFF: usize = 32;

fn karatsuba_sqr_core(
    buf: &[u64],
    half_len: usize,
    out: &mut [u64],
    cross: &mut [u64],
    scratch: &mut [u64],
) -> u64 {
    let (x0, x1) = buf.split_at(half_len);

    {
        let sum = &mut out[..=half_len];
        sum[..half_len].copy_from_slice(x0);
        let mut sum_len = half_len;
        if acc(&mut sum[..half_len], x1, 0) {
            sum[half_len] = 1;
            sum_len += 1;
        }

        karatsuba_sqr_alg(&sum[..sum_len], cross, scratch);
    }

    out.fill(0);
    let (z0, z2) = out.split_at_mut(2 * half_len);

    karatsuba_sqr_alg(x0, z0, scratch);
    let mut c = karatsuba_sqr_alg(x1, z2, scratch);

    acc(cross, z0, 1);
    acc(cross, z2, 1);
    if c > 0 {
        acc(&mut cross[z2.len()..], &[c], 1);
    }
    if acc(&mut out[half_len..], &cross, 0) {
        c += 1;
    }

    return c;
}

fn karatsuba_sqr_alg(buf: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    if buf.len() <= KARATSUBA_SQR_CUTOFF {
        return sqr_buf(buf, out);
    }
    let half = (buf.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_sqr_core(buf, half, out, cross, scratch)
}

fn find_karatsuba_sqr_scratch_size(mut n: usize) -> usize {
    let mut total = 0;
    while n > KARATSUBA_SQR_CUTOFF {
        let half = (n + 1) / 2;
        total += 2 * half + 1;
        n = half + 1;
    }
    total
}

pub(crate) fn sqr_vec(buf: &[u64]) -> (Vec<u64>, u64) {
    let mut out = vec![0_u64; 2 * buf.len() - 1];
    let c = if buf.len() <= KARATSUBA_SQR_CUTOFF {
        sqr_buf(buf, &mut out)
    } else {
        let mut scratch = vec![0_u64; find_karatsuba_sqr_scratch_size(buf.len())];
        let half_len = (buf.len() + 1) / 2;
        let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
        karatsuba_sqr_core(buf, half_len, &mut out, cross, rest)
    };
    return (out, c);
}

pub(crate) fn sqr_arr<const N: usize>(buf: &[u64]) -> Result<([u64; N], u64), ()> {
    if 2 * buf.len() - 1 > N {
        return Err(());
    }

    let mut out = [0_u64; N];
    let c = if buf.len() <= KARATSUBA_SQR_CUTOFF {
        sqr_buf(buf, &mut out)
    } else {
        let mut scratch = [0_u64; N];
        let scratch_sz = find_karatsuba_sqr_scratch_size(buf.len());
        let half_len = (buf.len() + 1) / 2;
        if scratch_sz > N {
            let mut cross = [0; N];
            karatsuba_sqr_core(buf, half_len, &mut out, &mut cross, &mut scratch)
        } else {
            let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
            karatsuba_sqr_core(buf, half_len, &mut out, cross, rest)
        }
    };
    return Ok((out, c));
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shr_carry_aarch64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "lsl {tmp}, {e}, {ms:x}", // put the last bits of the e into tmp
        "lsr {e}, {e}, {r:x}", // shift e by rem
        "orr {e}, {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg) sh,
        ms = in(reg) mv_sz,
        tmp = out(reg) _,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shr_carry_x86_64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "mov {tmp}, {e}", // put e into tmp
        "mov cl , {ms}", // move into cl reg
        "shl {tmp}, cl", // get last digits of e into tmp
        "mov cl , {r}", // move into cl reg
        "shr {e}, cl", // shift e by rem
        "or {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg_byte) sh,
        ms = in(reg_byte) mv_sz,
        tmp = out(reg) _,
        out("cl") _ ,
        options(nostack)
    );
}

unsafe fn shr_carry(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    #[cfg(target_arch = "aarch64")]
    shr_carry_aarch64(e, c, sh, mv_sz);

    #[cfg(target_arch = "x86_64")]
    shr_carry_x86_64(e, c, sh, mv_sz);
}

pub(crate) fn shr_buf(buf: &mut [u64], sh: u8) -> u64 {
    if sh == 0 {
        return 0;
    }

    assert!(sh < 64);
    let mv_sz = 64 - sh;
    unsafe {
        let mut c = 0;
        for e in buf.iter_mut().rev() {
            shr_carry_aarch64(e, &mut c, sh, mv_sz);
        }
        return c;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shl_carry_aarch64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "lsr {tmp}, {e}, {ms:x}", // put the last bits of the e into tmp
        "lsl {e}, {e}, {r:x}", // shift e by rem
        "orr {e}, {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg) sh,
        ms = in(reg) mv_sz,
        tmp = out(reg) _,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shl_carry_x86_64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "mov {tmp}, {e}", // put e into tmp
        "mov cl , {ms}", // move into cl reg
        "shr {tmp}, cl", // get last digits of e into tmp
        "mov cl , {r}", // move into cl reg
        "shl {e}, cl", // shift e by rem
        "or {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg_byte) sh,
        ms = in(reg_byte) mv_sz,
        tmp = out(reg) _,
        out("cl") _ ,
        options(nostack)
    );
}

unsafe fn shl_carry(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    #[cfg(target_arch = "aarch64")]
    shl_carry_aarch64(e, c, sh, mv_sz);

    #[cfg(target_arch = "x86_64")]
    shl_carry_x86_64(e, c, sh, mv_sz);
}

pub(crate) fn shl_buf(buf: &mut [u64], sh: u8) -> u64 {
    if sh == 0 {
        return 0;
    }

    assert!(sh < 64);
    let mv_sz = 64 - sh;
    unsafe {
        let mut c = 0;
        for e in buf.iter_mut() {
            shl_carry(e, &mut c, sh, mv_sz);
        }
        return c;
    }
}

pub(crate) fn shl_vec(vec: &mut Vec<u64>, sh: u64) {
    let div = (sh / 64) as usize;
    if div > 0 {
        let len = vec.len();
        vec.resize(len + div, 0);
        vec.copy_within(0..len, div as usize);
        vec[..div].fill(0);
    }

    let rem = (sh % 64) as u8;
    let c = shl_buf(vec, rem);
    if c > 0 {
        vec.push(c);
    }
}

pub(crate) fn shr_vec(vec: &mut Vec<u64>, sh: u64) {
    let div = (sh / 64) as usize;
    if div > 0 {
        if div < vec.len() {
            vec.drain(0..div);
        } else {
            vec.clear();
            return;
        }
    }

    let rem = (sh % 64) as u8;
    shr_buf(vec, rem);
    if *vec.last().unwrap() == 0 {
        vec.pop();
    }
}

pub(crate) fn shl_arr(buf: &mut [u64], sh: u64) {
    let div = (sh / 64) as usize;
    if div >= buf.len() {
        buf.fill(0);
        return;
    }
    if div > 0 {
        buf.copy_within(0..(buf.len() - div), div);
        buf[..div].fill(0);
    }

    let rem = (sh % 64) as u8;
    shl_buf(&mut buf[div..], rem);
}

pub(crate) fn shr_arr(buf: &mut [u64], sh: u64) {
    let div = (sh / 64) as usize;
    if div >= buf.len() {
        buf.fill(0);
        return;
    }
    if div > 0 {
        buf.copy_within(div..buf.len(), 0);
        buf[(buf.len() - div)..].fill(0);
    }

    let rem = (sh % 64) as u8;
    shl_buf(&mut buf[..(buf.len() - div)], rem);
}

pub(crate) fn div_prim(buf: &mut [u64], prim: u64) -> u64 {
    let prim_u128 = prim as u128;
    let mut rem: u128 = 0;
    let mut val: u128 = 0;
    for e in buf.iter_mut().rev() {
        val = (rem << 64) | (*e as u128);
        rem = val % prim_u128;
        *e = (val / prim_u128) as u64;
    }

    return rem as u64;
}

fn sub_mul_of(win: &mut [u64], of: &mut u64, d: &[u64], q: u64) -> bool {
    let q_u128 = q as u128;
    let mut carry = 0_u128;

    unsafe {
        let mut c = 1_u8;
        for (w, &d) in win.iter_mut().zip(d) {
            let val = q_u128 * (d as u128) + carry;
            carry = val >> 64;

            add_with_carry(w, !(val as u64), &mut c);
        }
        add_with_carry(of, !(carry as u64), &mut c);
        return c == 0;
    }
}

fn add_of(win: &mut [u64], of: &mut u64, d: &[u64]) {
    unsafe {
        let mut c = 0_u8;
        for (w, &d) in win.iter_mut().zip(d) {
            add_with_carry(w, d, &mut c);
        }
        *of = of.wrapping_add(c as u64);
    }
}

fn knuth_est(win: &mut [u64], of: &mut u64, d: &[u64], d1: u128, d0: u128, dfull: u128) -> u64 {
    let win_last = *win.last().unwrap() as u128;
    let of_u128 = *of as u128;
    let val = (of_u128 << 64) | win_last;
    let (mut qhat, rhat) = if of_u128 >= d1 {
        (u64::MAX, val - (u64::MAX as u128) * d1)
    } else {
        ((val / d1) as u64, val % d1)
    };

    if rhat < (1u128 << 64) {
        let a = (qhat as u128) * d0;
        let b = (rhat << 64) | (win[win.len() - 2] as u128);
        if a > b {
            qhat -= if (a - b) > dfull { 2 } else { 1 };
        }
    }

    if sub_mul_of(win, of, d, qhat) {
        qhat -= 1;
        add_of(win, of, d);
    }
    return qhat;
}

pub(crate) fn div_buf_of(n: &mut [u64], of: &mut u64, d: &[u64], out: &mut [u64]) {
    let d_len = d.len();
    let n_len = n.len();

    let d1 = d[d_len - 1] as u128;
    let d0 = d[d_len - 2] as u128;
    let dfull = ((d1 as u128) << 64) | (d0 as u128);

    let q_len = n_len - d_len;

    out[q_len] = knuth_est(&mut n[q_len..], of, d, d1, d0, dfull);
    for i in (0..q_len).rev() {
        out[i] = knuth_est(&mut n[i..i + d_len], &mut n[i + d_len], d, d1, d0, dfull)
    }
}

fn div_3_2(n: &mut [u64], d: &[u64], half_len: usize, q: &mut [u64], scratch: &mut [u64]) {
    let (d_lo, d_hi) = d.split_at(half_len);
    if cmp_buf(&n[2 * half_len..], d_hi) == std::cmp::Ordering::Less {
        div_2_1(&mut n[half_len..], d_hi, q, scratch);
    } else {
        q.fill(u64::MAX);
        acc(&mut n[2 * half_len..], d_hi, 1);
        acc(&mut n[half_len..], d_hi, 0);
    }

    let (m, k_scratch) = scratch.split_at_mut(q.len() + half_len);
    karatsuba_alg(q, d_lo, m, k_scratch);

    if acc(n, &m, 1) {
        dec(q);
        if !acc(n, d, 0) {
            dec(q);
            acc(n, d, 0);
        }
    }
}

fn div_3_2_static<const N: usize>(
    n: &mut [u64],
    d: &[u64],
    half_len: usize,
    q: &mut [u64],
    m: &mut [u64],
    scratch: &mut [u64],
) {
    let (d_lo, d_hi) = d.split_at(half_len);
    if cmp_buf(&n[2 * half_len..], d_hi) == std::cmp::Ordering::Less {
        div_2_1(&mut n[half_len..], d_hi, q, scratch);
    } else {
        q.fill(u64::MAX);
        acc(&mut n[2 * half_len..], d_hi, 1);
        acc(&mut n[half_len..], d_hi, 0);
    }

    karatsuba_alg(q, d_lo, m, scratch);

    if acc(n, &m, 1) {
        dec(q);
        if !acc(n, d, 0) {
            dec(q);
            acc(n, d, 0);
        }
    }
}

const BZ_CUTOFF: usize = 64;

fn div_2_1(n: &mut [u64], d: &[u64], q: &mut [u64], scratch: &mut [u64]) {
    let dlen = d.len();
    if dlen <= BZ_CUTOFF {
        div_buf_of(n, &mut 0, d, q);
        return;
    }

    let half_len = (dlen + 1) / 2;
    let (q_lo, q_hi) = q.split_at_mut(half_len);
    div_3_2(&mut n[half_len..], d, half_len, q_hi, scratch);
    div_3_2(&mut n[0..3 * half_len], d, half_len, q_lo, scratch);
}

fn div_2_1_static<const N: usize>(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    m: &mut [u64],
    scratch: &mut [u64],
) {
    let dlen = d.len();
    if dlen <= BZ_CUTOFF {
        div_buf_of(n, &mut 0, d, q);
        return;
    }

    let half_len = (dlen + 1) / 2;
    let (q_lo, q_hi) = q.split_at_mut(half_len);
    div_3_2_static(&mut n[half_len..], d, half_len, q_hi, m, scratch);
    div_3_2_static(&mut n[0..3 * half_len], d, half_len, q_lo, m, scratch);
}

fn find_bz_scratch_size(d: usize) -> usize {
    if d <= BZ_CUTOFF {
        return 0;
    }
    let half = (d + 1) / 2;
    2 * half - 1 + find_karatsuba_scratch_size(half, half)
}

fn bz_div_init(n: &mut [u64], d: &mut [u64], out: &mut [u64]) -> Option<(usize, u8)> {
    let dlen = d.len();
    let nlen = n.len();

    if dlen == 1 {
        out[..nlen].copy_from_slice(n);
        let rem = div_prim(out, d[0]);
        n.fill(0);
        n[0] = rem;
        return None;
    }

    let last_lz = d[dlen - 1].leading_zeros() as u8;
    shl_buf(d, last_lz);
    let mut last_n = shl_buf(n, last_lz);

    let t = (nlen + 1) / dlen - 1;
    let init_idx = dlen * t;
    div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..]);
    return Some((t, last_lz));
}

fn bz_div_alg(
    n: &mut [u64],
    d: &mut [u64],
    out: &mut [u64],
    t: usize,
    mut div_fn: impl FnMut(&mut [u64], &[u64], &mut [u64]),
) {
    let dlen = d.len();
    for i in (0..t).rev() {
        let idx = dlen * i;
        div_fn(&mut n[idx..idx + 2 * dlen], d, &mut out[idx..idx + dlen]);
    }
}

pub(crate) fn div_vec(n: &mut [u64], d: &mut [u64]) -> Vec<u64> {
    if d.len() == 0 {
        panic!("division by zero");
    }
    if n.len() < d.len() {
        return vec![];
    }
    let mut out = vec![0_u64; n.len() + 1 - d.len()];
    if let Some((t, sh)) = bz_div_init(n, d, &mut out) {
        if t > 0 {
            let size = find_bz_scratch_size(d.len());
            let mut scratch = vec![0_u64; size];
            bz_div_alg(n, d, &mut out, t, |n, d, q| div_2_1(n, d, q, &mut scratch));
        }
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
    return out;
}

pub(crate) fn div_arr<const N: usize>(n: &mut [u64], d: &mut [u64]) -> [u64; N] {
    let mut out = [0_u64; N];
    if let Some((t, sh)) = bz_div_init(n, d, &mut out) {
        if t > 0 {
            let size = find_bz_scratch_size(d.len());
            let mut scratch = [0_u64; N];
            if size > N {
                let mut m = [0_u64; N];
                bz_div_alg(n, d, &mut out, t, |n, d, q| {
                    div_2_1_static(n, d, q, &mut m, &mut scratch)
                });
            } else {
                bz_div_alg(n, d, &mut out, t, |n, d, q| div_2_1(n, d, q, &mut scratch));
            }
        }
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
    return out;
}

pub(crate) fn powi_vec(buf: &[u64], mut pow: u64) -> Vec<u64> {
    if pow == 0 {
        return vec![1];
    }

    let l = buf_len(buf);
    let log2 = (l - 1) * 64 + buf[l - 1].ilog2() as usize;
    let tmp_sz = 1 + (log2 * pow as usize) / 64;
    let scratch_sz = std::cmp::max(
        find_karatsuba_sqr_scratch_size(tmp_sz / 2),
        find_karatsuba_scratch_size(tmp_sz, l),
    );
    let mut tmp = vec![0_u64; tmp_sz];
    let mut out = vec![0_u64; tmp_sz];
    let mut scratch = vec![0_u64; scratch_sz];

    let mut len = buf.len();
    out[..len].copy_from_slice(buf);
    let mut io = true;

    let log = pow.ilog2() as usize;
    for i in (0..log).rev() {
        let mut sqr_len = 2 * len - 1;
        let (src, dst): (&[u64], &mut [u64]) = if io {
            (&out, &mut tmp)
        } else {
            (&tmp, &mut out)
        };
        let sqr_c = karatsuba_sqr_alg(&src[..len], &mut dst[..sqr_len], &mut scratch);
        len = sqr_len;
        if sqr_c > 0 {
            dst[len] = sqr_c;
            len += 1;
        }
        io = !io;

        if (pow >> i) & 1 == 1 {
            let mul_len = len + buf.len() - 1;
            let (src, dst): (&[u64], &mut [u64]) = if io {
                (&out, &mut tmp)
            } else {
                (&tmp, &mut out)
            };

            let mul_c = karatsuba_alg(&src[..len], &buf, &mut dst[..mul_len], &mut scratch);
            len = mul_len;
            if mul_c > 0 {
                dst[len] = mul_c;
                len += 1
            }
            io = !io;
        }
    }

    if !io {
        out[..len].copy_from_slice(&tmp[..len]);
    }
    trim_lz(&mut out);
    return out;
}

pub(crate) fn powi_arr<const N: usize>(buf: &[u64], pow: u64) -> Result<[u64; N], ()> {
    let l = buf_len(buf);
    let log2 = (l - 1) * 64 + buf[l - 1].ilog2() as usize;
    let tmp_sz = (log2 * pow as usize) / 64;
    if tmp_sz > N {
        return Err(());
    }

    let mut out = [0; N];
    let mut tmp = [0; N];
    let mut scratch = [0; N];

    let buf_len = buf_len(buf);
    let mut len = buf_len;
    out[..len].copy_from_slice(&buf[..len]);
    let mut io = true;

    let log = pow.ilog2();
    for i in (1..log).rev() {
        let sqr_len = 2 * len - 1;
        let (src, dst): (&[u64], &mut [u64]) = if io {
            (&out, &mut tmp)
        } else {
            (&tmp, &mut out)
        };

        let sqr_c = karatsuba_sqr_alg(&src[..len], &mut dst[..sqr_len], &mut scratch);
        len = sqr_len;
        if sqr_c > 0 {
            dst[len] = sqr_c;
            len += 1;
        }
        io = !io;
        if (pow >> i) & 1 == 1 {
            let mul_len = len + buf_len - 1;
            let (src, dst): (&[u64], &mut [u64]) = if io {
                (&out, &mut tmp)
            } else {
                (&tmp, &mut out)
            };

            let mul_c = karatsuba_alg(
                &src[..len],
                &buf[..buf_len],
                &mut dst[..mul_len],
                &mut scratch,
            );
            len = mul_len;
            if mul_c > 0 {
                dst[len] = mul_c;
                len += 1
            }
            io = !io;
        }
    }

    let sqr_len = 2 * len - 1;
    if sqr_len > N {
        return Err(());
    }

    let (src, dst): (&[u64], &mut [u64]) = if io {
        (&out, &mut tmp)
    } else {
        (&tmp, &mut out)
    };

    let sqr_c = if find_karatsuba_sqr_scratch_size(len) > N {
        let mut cross = [0; N];
        karatsuba_sqr_core(
            &src[..len],
            (len + 1) / 2,
            &mut dst[..sqr_len],
            &mut cross,
            &mut scratch,
        )
    } else {
        karatsuba_sqr_alg(&src[..len], &mut dst[..sqr_len], &mut scratch)
    };

    len = sqr_len;
    if sqr_c > 0 {
        if len == N {
            return Err(());
        }
        dst[len] = sqr_c;
        len += 1;
    }
    io = !io;

    if pow & 1 == 1 {
        let mul_len = len + buf_len - 1;
        if mul_len > N {
            return Err(());
        }

        let (src, dst): (&[u64], &mut [u64]) = if io {
            (&out, &mut tmp)
        } else {
            (&tmp, &mut out)
        };

        let mul_c = if find_karatsuba_scratch_size(len, buf_len) > N {
            let mut cross = [0; N];
            karatsuba_core(
                &src[..len],
                &buf[..buf_len],
                (len + 1) / 2,
                &mut dst[..mul_len],
                &mut cross,
                &mut scratch,
            )
        } else {
            karatsuba_alg(
                &src[..len],
                &buf[..buf_len],
                &mut dst[..mul_len],
                &mut scratch,
            )
        };

        len = mul_len;
        if mul_c > 0 {
            if len == N {
                return Err(());
            }
            dst[len] = mul_c;
            len += 1
        }
        io = !io;
    }

    if !io {
        out.copy_from_slice(&tmp);
    }
    return Ok(out);
}

fn short_mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let a_len = a.len() - 1;
    let b_len = b.len() - 1;
    let d = (a_len + b_len + 1).saturating_sub(out.len());

    let mask = u64::MAX as u128;
    let mut carry: u128 = 0;
    for n in d..(d + out.len()) {
        let mut term: u128 = carry;
        carry = 0;
        let mi = n.saturating_sub(b_len);
        let mf = n.min(a_len);
        for m in mi..=mf {
            let val = (a[m] as u128) * (b[n - m] as u128);
            term += val & mask;
            carry += val >> 64;
        }
        carry += term >> 64;
        out[n - d] = term as u64;
    }

    return carry as u64;
}

const SHORT_CUTOFF: usize = 32;

pub(crate) fn short_mul_vec(a: &[u64], b: &[u64], prec: usize) -> (Vec<u64>, u64) {
    let p = prec.min(a.len() + b.len() - 1);
    return if p <= SHORT_CUTOFF {
        let mut out = vec![0; p];
        let c = short_mul_buf(a, b, &mut out);
        (out, c)
    } else {
        let ap = a.len().min(p);
        let bp = b.len().min(p);
        let (mut out, c) = mul_vec(&a[a.len() - ap..], &b[b.len() - bp..]);
        out.drain(..out.len().saturating_sub(p));
        (out, c)
    };
}

fn short_sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let len = buf.len() - 1;
    let mask = u64::MAX as u128;
    let d = (2 * len + 1).saturating_sub(out.len());

    let mut carry = if d == 0 {
        let init_term = (buf[0] as u128) * (buf[0] as u128);
        out[0] = init_term as u64;
        init_term >> 64
    } else {
        0
    };
    for n in d.max(1)..(d + out.len()) {
        let mut term = carry;
        carry = 0;

        let mi = n.saturating_sub(len);
        let mf = ((n - 1) / 2).min(len);
        for m in mi..=mf {
            let val = (buf[m] as u128) * (buf[n - m] as u128);
            term += (val << 1) & mask;
            carry += val >> 63;
        }

        if n % 2 == 0 {
            let half_val = buf[n / 2] as u128;
            let val = half_val * half_val;
            term += val & mask;
            carry += val >> 64;
        }

        out[n - d] = term as u64;
        carry += term >> 64;
    }

    return carry as u64;
}

const SHORT_SQR_CUTOFF: usize = 32;

pub(crate) fn short_sqr_vec(buf: &[u64], prec: usize) -> (Vec<u64>, u64) {
    let p = prec.min(2 * buf.len() - 1);
    return if p <= SHORT_CUTOFF {
        let mut out = vec![0; p];
        let c = short_sqr_buf(buf, &mut out);
        (out, c)
    } else {
        let bufp = buf.len().min(p);
        let (mut out, c) = sqr_vec(&buf[buf.len() - bufp..]);
        out.drain(..out.len().saturating_sub(p));
        (out, c)
    };
}
