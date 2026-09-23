//! Standalone x86-64 sub_mul_of experiments; production source is unchanged.
//! Baseline instructions match div.rs, with the modified len correctly declared inout.
//! rustc --edition=2021 -O benches/probes/binom_submul.rs -o /tmp/binom_submul
//! taskset -c 2 /tmp/binom_submul
use std::{
    arch::asm,
    hint::black_box,
    time::{Duration, Instant},
};
type Kernel = unsafe fn(*mut u64, *mut u64, *const u64, u64, usize) -> bool;
#[inline(never)]
unsafe fn baseline(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let borrow: u8;
    asm!(
     "mov {mc}, 0",
     "mov {b}, 0",
     "2:",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win}], rax",
     "setc {b}",
     "lea {win}, [{win} + 8]",
     "lea {den}, [{den} + 8]",
     "dec {len}",
     "jnz 2b",
     "neg {b}",
     "sbb QWORD PTR [{ofp}], {mc}",
     "setc {b}",
     win=inout(reg)win=>_, den=inout(reg)d=>_, ofp=in(reg)of,
     len=inout(reg)len=>_, q=in(reg)q, b=out(reg_byte)borrow, mc=out(reg)_,
     out("rax")_, out("rdx")_, options(nostack),
    );
    borrow != 0
}

#[inline(never)]
unsafe fn merged(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let borrow: u8;
    asm!(
     "mov {mc}, 0",
     "2:",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win}], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "lea {win}, [{win} + 8]",
     "lea {den}, [{den} + 8]",
     "dec {len}",
     "jnz 2b",
     "sub QWORD PTR [{ofp}], {mc}",
     "setc {b}",
     win=inout(reg)win=>_, den=inout(reg)d=>_, ofp=in(reg)of,
     len=inout(reg)len=>_, q=in(reg)q, b=out(reg_byte)borrow, mc=out(reg)_,
     out("rax")_, out("rdx")_, options(nostack),
    );
    borrow != 0
}

#[inline(never)]
unsafe fn unroll2(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let borrow: u8;
    asm!(
     "mov {mc}, 0",
     "mov {b}, 0",
     "test {len}, 1",
     "jz 2f",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win}], rax",
     "setc {b}",
     "lea {win}, [{win} + 8]",
     "lea {den}, [{den} + 8]",
     "dec {len}",
     "2:",
     "test {len}, {len}",
     "jz 5f",
     "4:",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win}], rax",
     "setc {b}",
     "mov rax, [{den} + 8]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win} + 8], rax",
     "setc {b}",
     "lea {win}, [{win} + 16]",
     "lea {den}, [{den} + 16]",
     "sub {len}, 2",
     "jnz 4b",
     "5:",
     "neg {b}",
     "sbb QWORD PTR [{ofp}], {mc}",
     "setc {b}",
     win=inout(reg)win=>_, den=inout(reg)d=>_, ofp=in(reg)of,
     len=inout(reg)len=>_, q=in(reg)q, b=out(reg_byte)borrow, mc=out(reg)_,
     out("rax")_, out("rdx")_, options(nostack),
    );
    borrow != 0
}

#[inline(never)]
unsafe fn unroll4(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let borrow: u8;
    asm!(
     "mov {mc}, 0",
     "mov {b}, 0",
     "test {len}, 1",
     "jz 2f",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win}], rax",
     "setc {b}",
     "lea {win}, [{win} + 8]",
     "lea {den}, [{den} + 8]",
     "dec {len}",
     "2:",
     "test {len}, 2",
     "jz 3f",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win}], rax",
     "setc {b}",
     "mov rax, [{den} + 8]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win} + 8], rax",
     "setc {b}",
     "lea {win}, [{win} + 16]",
     "lea {den}, [{den} + 16]",
     "sub {len}, 2",
     "3:",
     "test {len}, {len}",
     "jz 5f",
     "4:",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win}], rax",
     "setc {b}",
     "mov rax, [{den} + 8]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win} + 8], rax",
     "setc {b}",
     "mov rax, [{den} + 16]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win} + 16], rax",
     "setc {b}",
     "mov rax, [{den} + 24]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "neg {b}",
     "sbb QWORD PTR [{win} + 24], rax",
     "setc {b}",
     "lea {win}, [{win} + 32]",
     "lea {den}, [{den} + 32]",
     "sub {len}, 4",
     "jnz 4b",
     "5:",
     "neg {b}",
     "sbb QWORD PTR [{ofp}], {mc}",
     "setc {b}",
     win=inout(reg)win=>_, den=inout(reg)d=>_, ofp=in(reg)of,
     len=inout(reg)len=>_, q=in(reg)q, b=out(reg_byte)borrow, mc=out(reg)_,
     out("rax")_, out("rdx")_, options(nostack),
    );
    borrow != 0
}

#[inline(never)]
unsafe fn merged2(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let borrow: u8;
    asm!(
     "mov {mc}, 0",
     "test {len}, 1",
     "jz 2f",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win}], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "lea {win}, [{win} + 8]",
     "lea {den}, [{den} + 8]",
     "dec {len}",
     "2:",
     "test {len}, {len}",
     "jz 5f",
     "4:",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win}], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "mov rax, [{den} + 8]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win} + 8], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "lea {win}, [{win} + 16]",
     "lea {den}, [{den} + 16]",
     "sub {len}, 2",
     "jnz 4b",
     "5:",
     "sub QWORD PTR [{ofp}], {mc}",
     "setc {b}",
     win=inout(reg)win=>_, den=inout(reg)d=>_, ofp=in(reg)of,
     len=inout(reg)len=>_, q=in(reg)q, b=out(reg_byte)borrow, mc=out(reg)_,
     out("rax")_, out("rdx")_, options(nostack),
    );
    borrow != 0
}

#[inline(never)]
unsafe fn merged4(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let borrow: u8;
    asm!(
     "mov {mc}, 0",
     "test {len}, 1",
     "jz 2f",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win}], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "lea {win}, [{win} + 8]",
     "lea {den}, [{den} + 8]",
     "dec {len}",
     "2:",
     "test {len}, 2",
     "jz 3f",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win}], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "mov rax, [{den} + 8]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win} + 8], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "lea {win}, [{win} + 16]",
     "lea {den}, [{den} + 16]",
     "sub {len}, 2",
     "3:",
     "test {len}, {len}",
     "jz 5f",
     "4:",
     "mov rax, [{den}]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win}], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "mov rax, [{den} + 8]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win} + 8], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "mov rax, [{den} + 16]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win} + 16], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "mov rax, [{den} + 24]",
     "mul {q}",
     "add rax, {mc}",
     "adc rdx, 0",
     "sub QWORD PTR [{win} + 24], rax",
     "adc rdx, 0",
     "mov {mc}, rdx",
     "lea {win}, [{win} + 32]",
     "lea {den}, [{den} + 32]",
     "sub {len}, 4",
     "jnz 4b",
     "5:",
     "sub QWORD PTR [{ofp}], {mc}",
     "setc {b}",
     win=inout(reg)win=>_, den=inout(reg)d=>_, ofp=in(reg)of,
     len=inout(reg)len=>_, q=in(reg)q, b=out(reg_byte)borrow, mc=out(reg)_,
     out("rax")_, out("rdx")_, options(nostack),
    );
    borrow != 0
}

fn reference(work: &mut [u64], d: &[u64], q: u64) -> bool {
    let mut carry = 0u64;
    for (w, &di) in work.iter_mut().zip(d) {
        let product = di as u128 * q as u128 + carry as u128;
        let (value, borrow) = w.overflowing_sub(product as u64);
        *w = value;
        carry = ((product >> 64) as u64).checked_add(borrow as u64).unwrap();
    }
    let (top, borrow) = work[d.len()].overflowing_sub(carry);
    work[d.len()] = top;
    borrow
}
fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
struct Case {
    d: Vec<u64>,
    w: Vec<u64>,
    q: u64,
}
fn cases(n: usize, count: usize, state: &mut u64) -> Vec<Case> {
    (0..count)
        .map(|i| {
            let mut d: Vec<_> = (0..n).map(|_| rng(state)).collect();
            let mut w: Vec<_> = (0..=n).map(|_| rng(state)).collect();
            let q = match i % 32 {
                0 => 0,
                1 => 1,
                2 => u64::MAX,
                3 => 1 << 63,
                _ => rng(state),
            };
            if i % 32 == 4 {
                d.fill(u64::MAX);
                w.fill(0);
            }
            if i % 32 == 5 {
                d.fill(u64::MAX);
                w.fill(u64::MAX);
            }
            if i % 32 == 6 {
                d.fill(0);
                w.fill(0);
            }
            Case { d, w, q }
        })
        .collect()
}
fn time(kernel: Kernel, inputs: &[Case], duration: Duration) -> f64 {
    let mut work = inputs[0].w.clone();
    let start = Instant::now();
    let mut count = 0u64;
    let mut sink = 0u64;
    loop {
        for case in inputs {
            let case = black_box(case);
            work.copy_from_slice(&case.w);
            let p = work.as_mut_ptr();
            let borrow = unsafe {
                kernel(
                    p,
                    p.add(case.d.len()),
                    case.d.as_ptr(),
                    case.q,
                    case.d.len(),
                )
            };
            sink ^= work[0] ^ work[case.d.len()] ^ borrow as u64;
        }
        count += inputs.len() as u64;
        if start.elapsed() >= duration {
            break;
        }
    }
    black_box(sink);
    start.elapsed().as_nanos() as f64 / count as f64
}
fn main() {
    let entries: [(&str, Kernel); 6] = [
        ("baseline", baseline),
        ("merged", merged),
        ("unroll2", unroll2),
        ("unroll4", unroll4),
        ("merged2", merged2),
        ("merged4", merged4),
    ];
    let mut state = 0x348734283473287u64;
    let mut checked = 0u64;
    for n in 1..=70 {
        let mut inputs = cases(n, 512, &mut state);
        for q in [0, 1, 1 << 63, u64::MAX] {
            for digit in [0, u64::MAX] {
                for word in [0, u64::MAX] {
                    for top in [0, u64::MAX] {
                        let mut w = vec![word; n + 1];
                        w[n] = top;
                        inputs.push(Case {
                            d: vec![digit; n],
                            w,
                            q,
                        });
                    }
                }
            }
        }
        for case in inputs {
            let mut expected = case.w.clone();
            let borrow = reference(&mut expected, &case.d, case.q);
            for &(name, kernel) in &entries {
                let mut work = case.w.clone();
                let p = work.as_mut_ptr();
                let got = unsafe { kernel(p, p.add(n), case.d.as_ptr(), case.q, n) };
                assert_eq!(work, expected, "{name} n={n}, q={}", case.q);
                assert_eq!(got, borrow, "{name} n={n}");
                checked += 1;
            }
        }
    }
    println!("checked={checked}");
    if std::env::args().any(|arg| arg == "--check-only") {
        return;
    }
    println!("limbs,variant,ns_per_call,change_pct");
    for n in [2, 3, 4, 8, 16, 32, 63] {
        let inputs = cases(n, 256, &mut state);
        let mut results: Vec<Vec<f64>> = (0..entries.len()).map(|_| Vec::new()).collect();
        for trial in 0..5 {
            for i in 0..entries.len() {
                let index = (i + trial) % entries.len();
                results[index].push(time(entries[index].1, &inputs, Duration::from_millis(80)));
            }
        }
        for r in &mut results {
            r.sort_by(f64::total_cmp);
        }
        for (i, (name, _)) in entries.iter().enumerate() {
            println!(
                "{n},{name},{:.3},{:.2}",
                results[i][2],
                100. * (results[i][2] / results[0][2] - 1.)
            );
        }
    }
}
