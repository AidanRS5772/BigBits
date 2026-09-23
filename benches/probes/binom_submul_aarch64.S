// Candidate only: syntax-checked with Clang, not timed or executed on AArch64.
// ABI: x0=win, x1=overflow limb, x2=divisor, x3=quotient, x4=length (>0).
// Return w0 = final borrow. No extra architecture extensions are required.
// clang --target=aarch64-linux-gnu -c benches/probes/binom_submul_aarch64.S -o /tmp/submul-arm.o
.text
.p2align 4
.global submul_merged_aarch64
.type submul_merged_aarch64, %function
submul_merged_aarch64:
    mov x5, xzr
.Lloop:
    ldr x9, [x2], #8
    ldr x8, [x0]
    mul x6, x9, x3
    umulh x7, x9, x3
    adds x6, x6, x5
    adc x7, x7, xzr
    subs x8, x8, x6
    cinc x5, x7, cc
    str x8, [x0], #8
    sub x4, x4, #1
    cbnz x4, .Lloop
    ldr x8, [x1]
    subs x8, x8, x5
    str x8, [x1]
    cset w0, cc
    ret
.size submul_merged_aarch64, .-submul_merged_aarch64
