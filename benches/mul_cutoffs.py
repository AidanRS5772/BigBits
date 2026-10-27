#!/usr/bin/env python3
"""Builds, runs and analyzes benches/mul_cutoffs.rs (see docs/mul_cutoff_tuning.md).

  python3 benches/mul_cutoffs.py build [--layouts plain,jcc,jcc_align] [--set NAME=VALUE ...]
  python3 benches/mul_cutoffs.py run FAMILY... [--reps 8] [--core 2|all] [--tag T] [-- bench args]
  python3 benches/mul_cutoffs.py analyze FAMILY... [--tag T] [--alpha 0.05] [--simple]

build compiles the bench under each code layout (fat-LTO release, one target dir each,
in parallel) and records the mod.rs constants it was built with. --set overrides a
mod.rs constant for that build only (the file is restored afterwards), e.g. to give
Karatsuba schoolbook leaves while its own cutoffs are measured. --check instead builds
once without LTO, for `target/mul_cutoffs/check/exe.txt <family> --check`, which runs
both sides once per input and compares them without timing. run starts reps
processes per layout and family, one at a time, pinned to one core (which also gives
rayon a one-thread pool) or, with --core all, unpinned with rayon's full pool, in a
random layout order per rep, and records the load average before each. analyze treats each process as one observation: per shape, an
exact one-sided Wilcoxon signed-rank test on the per-process medians of
ln(t_upper / t_lower), a Hodges-Lehmann ratio and its 95% CI, then each family's
cutoff rule. That full treatment is for nanosecond-scale shapes, which need many
processes (24). Microsecond-scale families need only a few processes (one per layout)
and --simple, which decides each shape by the mean over processes.

Only the standard library is used.
"""
import argparse, glob, json, math, os, random, re, shutil, subprocess, sys, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "target", "mul_cutoffs")
LAYOUTS = {
    "plain": "",
    "jcc": "-C llvm-args=-x86-branches-within-32B-boundaries",
    "jcc_align": "-C llvm-args=-x86-branches-within-32B-boundaries -C llvm-args=-align-loops=64",
}
FAMILIES = "prim chunk kara kfft kntt ksqr sqrfft sqrntt hi hisqr hirows mid midntt".split()


def consts_snapshot():
    text = open(os.path.join(ROOT, "src", "utils", "mod.rs")).read()
    return "\n".join(m.group(0) for m in re.finditer(r"pub const \w+: [\w0-9]+ = [^;]+;", text))


# --- build / run -------------------------------------------------------------

def cmd_build(args):
    mod_rs = os.path.join(ROOT, "src", "utils", "mod.rs")
    original = open(mod_rs).read()
    text = original
    for item in args.set:
        name, value = item.split("=", 1)
        pat = re.compile(rf"(pub const {re.escape(name)}: [\w0-9]+ = )[^;]+;")
        if not pat.search(text):
            sys.exit(f"no constant {name} in mod.rs")
        text = pat.sub(lambda m: m.group(1) + value + ";", text)
    try:
        if text != original:
            open(mod_rs, "w").write(text)
        build_layouts(args)
    finally:
        if text != original:
            open(mod_rs, "w").write(original)


def build_layouts(args):
    layouts = ["check"] if args.check else args.layouts.split(",")
    procs = {}
    for lay in layouts:
        env = dict(os.environ, RUSTFLAGS=LAYOUTS.get(lay, ""), CARGO_TARGET_DIR=os.path.join(BASE, lay))
        if args.check:
            env.update(CARGO_PROFILE_RELEASE_LTO="false", CARGO_PROFILE_RELEASE_CODEGEN_UNITS="16")
        cmd = ["cargo", "bench", "--profile", "release", "--bench", "mul_cutoffs", "--no-run",
               "--message-format=json-render-diagnostics"]
        procs[lay] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True)
    snap = consts_snapshot()
    for lay, p in procs.items():
        exe = None
        for line in p.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("reason") == "compiler-artifact" and msg.get("executable") and \
                    msg["target"]["name"] == "mul_cutoffs":
                exe = msg["executable"]
        if p.wait() != 0 or exe is None:
            sys.exit(f"build failed for {lay}")
        d = os.path.join(BASE, lay)
        open(os.path.join(d, "exe.txt"), "w").write(exe)
        open(os.path.join(d, "consts.txt"), "w").write(snap)
        print(f"{lay}: {exe}")


def cmd_run(args, extra):
    layouts = args.layouts.split(",")
    exes = {lay: open(os.path.join(BASE, lay, "exe.txt")).read().strip() for lay in layouts}
    out = os.path.join(BASE, "out", args.tag)
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(BASE, layouts[0], "consts.txt"), os.path.join(out, "consts.txt"))
    with open(os.path.join(out, "mode.txt"), "a") as f:
        f.write(f"{' '.join(args.families)}: core {args.core}, extra args {extra}\n")
    rng = random.Random(f"{args.tag}/{args.seed}")
    t_start = time.time()
    for rep in range(args.reps):
        order = layouts[:]
        rng.shuffle(order)
        for lay in order:
            for fam in args.families:
                d = os.path.join(out, fam)
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"{lay}_{rep + args.rep0}.csv")
                if os.path.exists(path) and not args.force:
                    continue
                seed = rng.getrandbits(48)
                load = open("/proc/loadavg").read().split()[0]
                t0 = time.time()
                pin = [] if args.core == "all" else ["taskset", "-c", args.core]
                cmd = pin + [exes[lay], fam, "--seed", str(seed)] + extra
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    sys.exit(f"{fam} {lay} rep {rep} failed:\n{res.stderr}")
                open(path + ".tmp", "w").write(res.stdout)
                os.replace(path + ".tmp", path)
                if res.stderr.strip():
                    open(path[:-4] + ".err", "w").write(res.stderr)
                    print(f"  {fam} {lay} rep {rep + args.rep0}: {len(res.stderr.splitlines())} stderr lines "
                          f"in {path[:-4]}.err", flush=True)
                with open(os.path.join(d, "load.txt"), "a") as f:
                    f.write(f"{lay} {rep + args.rep0} {load} {time.time() - t0:.1f}s\n")
                print(f"[{(time.time() - t_start) / 60:6.1f} min] {fam} {lay} rep {rep + args.rep0}: "
                      f"{time.time() - t0:.1f}s, load {load}", flush=True)


# --- statistics --------------------------------------------------------------

def wilcoxon(d):
    """Exact one-sided p-values (upper faster: d < 0, lower faster: d > 0)."""
    d = [x for x in d if x != 0.0]
    n = len(d)
    if n == 0:
        return 1.0, 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:  # average ranks over ties
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r2 = [int(round(2 * r)) for r in ranks]  # doubled ranks handle .5 ties
    tot = sum(r2)
    dist = [0] * (tot + 1)
    dist[0] = 1
    for r in r2:
        for s in range(tot, r - 1, -1):
            dist[s] += dist[s - r]
    total = 2 ** n
    wplus = sum(r for r, x in zip(r2, d) if x > 0)
    p_up = sum(dist[: wplus + 1]) / total  # P(W+ <= observed)
    p_lo = sum(dist[wplus:]) / total  # P(W+ >= observed)
    return p_up, p_lo


def hodges_lehmann(d, alpha=0.05):
    """Estimate and exact distribution-free CI for the center of d."""
    n = len(d)
    walsh = sorted((d[i] + d[j]) / 2 for i in range(n) for j in range(i, n))
    tot = n * (n + 1) // 2
    dist = [0] * (tot + 1)
    dist[0] = 1
    for r in range(1, n + 1):
        for s in range(tot, r - 1, -1):
            dist[s] += dist[s - r]
    total = 2 ** n
    cum, c = 0, 0
    while (cum + dist[c]) / total <= alpha / 2:
        cum += dist[c]
        c += 1
    m = len(walsh)
    mid = (walsh[(m - 1) // 2] + walsh[m // 2]) / 2
    return mid, walsh[min(c, m - 1)], walsh[max(m - 1 - c, 0)]


class Shape:
    __slots__ = ("path", "l", "s", "flip", "d", "tl", "tu", "p_up", "p_lo", "est", "lo", "hi", "v")


def load(tag, fam, alpha, simple=False):
    shapes = {}
    files = sorted(glob.glob(os.path.join(BASE, "out", tag, fam, "*.csv")))
    if not files:
        sys.exit(f"no data for {fam} under tag {tag}")
    for path in files:
        for line in open(path):
            f, p, l, s, flip, med, tl, tu = line.strip().split(",")
            key = (p, int(l), int(s))
            sh = shapes.get(key)
            if sh is None:
                sh = shapes[key] = Shape()
                sh.path, sh.l, sh.s, sh.flip = p, int(l), int(s), float(flip)
                sh.d, sh.tl, sh.tu = [], [], []
            sh.d.append(float(med))
            sh.tl.append(float(tl))
            sh.tu.append(float(tu))
    for sh in shapes.values():
        if simple:
            # Microsecond-scale shapes: the mean over processes decides, and lo/hi
            # are the extreme processes rather than a CI.
            sh.p_up = sh.p_lo = float("nan")
            sh.est, sh.lo, sh.hi = sum(sh.d) / len(sh.d), min(sh.d), max(sh.d)
            sh.v = "U" if sh.est < 0 else "L"
        else:
            sh.p_up, sh.p_lo = wilcoxon(sh.d)
            sh.est, sh.lo, sh.hi = hodges_lehmann(sh.d, alpha)
            sh.v = "U" if sh.p_up < alpha else "L" if sh.p_lo < alpha else "="
    return list(shapes.values()), len(files)


def ratio(x):
    return math.exp(x)


def fmt_shape(sh):
    dims = f"{sh.l}x{sh.s}" if sh.s else f"{sh.l}"
    return (f"{sh.path:>13} {dims:>11} flip {sh.flip:9.3f}  up/lo {ratio(sh.est):.4f} "
            f"[{ratio(sh.lo):.4f}, {ratio(sh.hi):.4f}]  p_up {sh.p_up:.1e} p_lo {sh.p_lo:.1e}  {sh.v}"
            f"  lo {median(sh.tl):.0f} ns")


def median(v):
    v = sorted(v)
    n = len(v)
    return (v[(n - 1) // 2] + v[n // 2]) / 2


def iut(shapes, upper_side=lambda sh, t: sh.flip > t):
    """The smallest threshold t such that every shape on the upper side of t is
    significantly faster on the upper side: the largest flip among shapes that
    aren't. Returns None when every shape is (the cutoff lies below the range)."""
    bad = [sh.flip for sh in shapes if sh.v != "U"]
    return max(bad) if bad else None


def regret(shapes, t):
    """Mean log-time lost against each shape's faster side, by HL estimate, when
    shapes with flip > t take the upper side."""
    loss = 0.0
    for sh in shapes:
        up = sh.flip > t
        loss += max(sh.est, 0.0) if up else max(-sh.est, 0.0)
    return loss / len(shapes)


def best_regret(shapes):
    cands = sorted({sh.flip for sh in shapes})
    cands = [cands[0] - 1e-9] + cands
    best = min(cands, key=lambda t: (regret(shapes, t), t))
    return best, regret(shapes, best)


def runs(shapes):
    """Verdicts as runs over sorted flip values: 'U 3-10, = 11-12, ...'."""
    out, cur = [], None
    for sh in sorted(shapes, key=lambda s: s.flip):
        x = f"{sh.flip:g}"
        if cur and cur[0] == sh.v:
            cur[2] = x
        else:
            cur = [sh.v, x, x]
            out.append(cur)
    return ", ".join(f"{v} {a}" + (f"-{b}" if a != b else "") for v, a, b in out)


def report_1d(name, shapes, op="<=", show=8):
    """A 1-D cutoff: the lower side runs while size <= C (op '<=') or size < C ('<')."""
    shapes = sorted(shapes, key=lambda s: s.flip)
    c = iut(shapes)
    if c is None:
        rec = f"below {shapes[0].flip:g} (upper side significant everywhere)"
        val = None
    elif c >= shapes[-1].flip:
        rec = f"at or above {shapes[-1].flip:g} (upper side not significant at the largest size)"
        val = None
    else:
        val = int(c) if op == "<=" else int(c) + 1
        rec = f"{val}"
    missed = [sh for sh in shapes if sh.v == "U" and c is not None and sh.flip <= c]
    print(f"  {name}: cutoff {rec}")
    print(f"    verdicts: {runs(shapes)}")
    if missed:
        print(f"    upper side significant below the cutoff at: {', '.join(f'{sh.flip:g}' for sh in missed)}")
    if c is not None:
        near = [sh for sh in shapes if abs(sh.flip - c) <= show * max(1, step_of(shapes))]
        for sh in near:
            print("     " + fmt_shape(sh))
    return val


def step_of(shapes):
    f = sorted({sh.flip for sh in shapes})
    return min((b - a for a, b in zip(f, f[1:])), default=1)


def report_param(name, shapes, fmt="{:.4f}", upper_name="upper"):
    """A parametric boundary: shapes with flip > t take the upper side."""
    c = iut(shapes)
    best, best_loss = best_regret(shapes)
    loss = regret(shapes, c if c is not None else min(sh.flip for sh in shapes) - 1)
    missed = sorted([sh for sh in shapes if sh.v == "U" and c is not None and sh.flip <= c],
                    key=lambda s: -s.flip)
    cnt = defaultdict(int)
    for sh in shapes:
        cnt[sh.v] += 1
    print(f"  {name}: {len(shapes)} shapes ({cnt['U']} U, {cnt['=']} =, {cnt['L']} L)")
    print(f"    rule: threshold {fmt.format(c) if c is not None else 'below range'}"
          f" (mean regret {loss:.4f}); regret-minimizing {fmt.format(best)} ({best_loss:.4f})")
    if c is not None:
        hold = [sh for sh in shapes if sh.v != "U" and sh.flip == c]
        for sh in hold:
            print("    set by " + fmt_shape(sh))
    print(f"    {len(missed)} shapes significantly faster on the {upper_name} side kept on the lower side"
          + (f"; worst: " if missed else ""))
    for sh in sorted(missed, key=lambda s: s.est)[:12]:
        print("     " + fmt_shape(sh))
    return c


def by_path(shapes, pred):
    return [sh for sh in shapes if pred(sh.path)]


def analyze(fam, tag, alpha, simple=False):
    shapes, files = load(tag, fam, alpha, simple)
    print(f"\n### {fam} ({files} processes, tag {tag})")
    paths = sorted({sh.path for sh in shapes})
    if fam == "prim":
        for p in paths:
            sub = by_path(shapes, lambda x: x == p)
            print(f"  {p}: mul_buf vs the Prim arm by l: {runs(sub)}")
            for sh in sorted(sub, key=lambda s: s.flip):
                if sh.l in (3, 4, 8, 16, 32, 64, 128, 256):
                    print("     " + fmt_shape(sh))
    elif fam == "chunk":
        for p in paths:
            report_1d(f"{p} (s)", by_path(shapes, lambda x: x == p), show=2)
        report_1d("dyn, all ratios", by_path(shapes, lambda x: not x.startswith("static")), show=0)
        report_1d("static, all ratios", by_path(shapes, lambda x: x.startswith("static")), show=0)
        report_1d("CHUNKING_KARATSUBA_CUTOFF (joint)", shapes, show=0)
    elif fam == "kara":
        for p in paths:
            sub = by_path(shapes, lambda x: x == p)
            report_param(f"{p} KARATSUBA_CUTOFF c", sub, "{:.3f}", "Karatsuba")
            sq = [sh for sh in sub if sh.l == sh.s]
            if sq:
                report_1d(f"{p} squares only (by c)", sq, show=0)
        report_param("KARATSUBA_CUTOFF c (joint)", shapes, "{:.3f}", "Karatsuba")
    elif fam in ("kfft", "kntt"):
        names = {"kfft": ("FFT_KARATSUBA_CUTOFF", "FFT_CHUNKING_KARATSUBA_CUTOFF"),
                 "kntt": ("NTT_KARATSUBA_CUTOFF", "NTT_CHUNKING_KARATSUBA_CUTOFF")}[fam]
        for p, n in (("reg", names[0]), ("chunk", names[1])):
            sub = by_path(shapes, lambda x: x == p)
            if sub:
                report_param(f"{n} (cost ratio)", sub, "{:.4f}", "transform")
                sq = [sh for sh in sub if sh.l == sh.s]
                if sq:
                    report_1d(f"{n} squares only (by ratio)", sq, show=0)
    elif fam == "ksqr":
        for p in paths:
            report_1d(f"{p} KARATSUBA_SQR_CUTOFF", by_path(shapes, lambda x: x == p))
        report_1d("KARATSUBA_SQR_CUTOFF (joint)", shapes, show=0)
    elif fam == "sqrfft":
        report_1d("school -> FFT", by_path(shapes, lambda x: x == "school"))
        report_1d("Karatsuba -> FFT (FFT_SQR_CUTOFF with a dyn Karatsuba tier)",
                  by_path(shapes, lambda x: x == "kara"))
    elif fam == "sqrntt":
        report_1d("STATIC_NTT_SQR_CUTOFF", shapes)
    elif fam in ("hi", "hisqr"):
        name = "PARTIAL_MUL_CUTOFF" if fam == "hi" else "PARTIAL_SQR_CUTOFF"
        for p in paths:
            report_1d(f"{p} {name}", by_path(shapes, lambda x: x == p), show=2)
        report_1d(f"dyn {name} (joint)", by_path(shapes, lambda x: x.startswith("dyn")), show=0)
        report_1d(f"static {name} (joint)", by_path(shapes, lambda x: x.startswith("static")), show=0)
        report_1d(f"{name} (all paths)", shapes, show=0)
    elif fam == "hirows":
        report_1d("HI_MUL_ROW_CUTOFF", by_path(shapes, lambda x: x == "mul"))
        report_1d("HI_SQR_ROW_CUTOFF", by_path(shapes, lambda x: x == "sqr"))
    elif fam == "mid":
        report_1d("FFT_MID_CUTOFF", shapes, op="<")
    elif fam == "midntt":
        report_1d("NTT_MID_CUTOFF", shapes, op="<")


def cmd_analyze(args):
    for fam in args.families:
        analyze(fam, args.tag, args.alpha, args.simple)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--layouts", default=",".join(LAYOUTS))
    b.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    b.add_argument("--check", action="store_true",
                   help="one fast non-LTO build (target/mul_cutoffs/check) for the bench's --check mode")
    r = sub.add_parser("run")
    r.add_argument("families", nargs="+", choices=FAMILIES)
    r.add_argument("--reps", type=int, default=8)
    r.add_argument("--rep0", type=int, default=0, help="first replicate index (to add more later)")
    r.add_argument("--layouts", default=",".join(LAYOUTS))
    r.add_argument("--core", default="2",
                   help="core to pin to, or 'all' to leave the process unpinned with rayon's full pool")
    r.add_argument("--tag", default="default")
    r.add_argument("--seed", default="0")
    r.add_argument("--force", action="store_true", help="rerun replicates that already have output")
    a = sub.add_parser("analyze")
    a.add_argument("families", nargs="+", choices=FAMILIES)
    a.add_argument("--tag", default="default")
    a.add_argument("--alpha", type=float, default=0.05)
    a.add_argument("--simple", action="store_true",
                   help="decide each shape by the mean over processes (for microsecond-scale families)")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "run":
        cmd_run(args, extra)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
