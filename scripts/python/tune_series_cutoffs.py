#!/usr/bin/env python3
"""Reproducible complete-series measurements and the dispatch cost model's fit.

  python3 scripts/python/tune_series_cutoffs.py run --tag train --reps 3 -- --threads 1
  python3 scripts/python/tune_series_cutoffs.py fit 'target/series_cutoffs/train/*.csv'
  python3 scripts/python/tune_series_cutoffs.py report 'target/series_cutoffs/heldout/*.csv'
  python3 scripts/python/tune_series_cutoffs.py blocks 'target/series_cutoffs/blocks/*.csv'

The model lives in series.rs: the bench prints each row's work vector (w_* columns), and
the fit only finds the ns per unit of each column, by non-negative least squares on
relative error. Raw measurements, metadata and per-process medians stay under target/.
"""
import argparse, collections, csv, glob, hashlib, itertools, json, math, os, pathlib, re, statistics, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=ROOT/'target/series_cutoffs'
ALGS=('Horner','Rect','BinSplit')
WORK=('div','scalar','product','term','call','alloc','fill')
CONSTS=('SERIES_NS_DIV','SERIES_NS_SCALAR','SERIES_NS_PRODUCT','SERIES_NS_TERM','SERIES_NS_CALL','SERIES_NS_ALLOC','SERIES_NS_FILL')
KEY=('mode','family','shape','wide','len','capacity','a','t','k','zeros','xl')

def run(args, extra):
    dest=BASE/args.tag
    if dest.exists() and any(dest.glob('process*.csv')):
        raise SystemExit(f'{dest} already has measurements; use a new tag')
    dest.mkdir(parents=True,exist_ok=True)
    cmd=['cargo','bench','--profile','release','--bench','series_cutoffs','--no-run','--message-format=json']
    build=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True)
    exe=next(json.loads(line)['executable'] for line in build.stdout.splitlines() if json.loads(line).get('executable') and json.loads(line).get('target',{}).get('name')=='series_cutoffs')
    metadata={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'rustc':subprocess.check_output(['rustc','-Vv'],text=True),'command':extra,'platform':os.uname()._asdict() if hasattr(os.uname(),'_asdict') else list(os.uname()),'environment':{k:v for k,v in os.environ.items() if k in ('RUSTFLAGS','RAYON_NUM_THREADS')},'constants':(ROOT/'src/utils/mod.rs').read_text()}
    metadata['executable_sha256']=hashlib.sha256(pathlib.Path(exe).read_bytes()).hexdigest()
    metadata['sources']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('benches/series_cutoffs.rs','src/utils/series.rs','src/utils/mod.rs','Cargo.toml')}
    (dest/'harness.rs').write_text((ROOT/'benches/series_cutoffs.rs').read_text())
    (dest/'analysis.py').write_text(pathlib.Path(__file__).read_text())
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for rep in range(args.reps):
        path=dest/f'process{rep}.csv'
        if path.exists(): raise SystemExit(f'{path} exists; use a new tag')
        start=time.monotonic()
        with path.open('w') as out, path.with_suffix('.log').open('w') as err:
            subprocess.run([exe,'--seed',str(71+rep*1009),*extra],cwd=ROOT,stdout=out,stderr=err,check=True)
        path.with_suffix('.done').write_text(str(time.monotonic()-start)+' seconds\n')
        print(path,flush=True)

def load(paths):
    """Rows of (fields, {alg: median ns}, {alg: work vector or None}, {alg: block length})."""
    groups=collections.defaultdict(lambda:collections.defaultdict(list)); fields={}; works={}; blocks={}
    for pattern in paths:
        for path in glob.glob(pattern):
            local=collections.defaultdict(lambda:collections.defaultdict(list))
            with open(path) as f:
                for r in csv.DictReader(f):
                    key=tuple(r[k] for k in KEY)
                    local[key][r['alg']].append(float(r['ns'])); fields[key]=r
                    if 'w_div' in r: works[key,r['alg']]=tuple(float(r['w_'+w]) for w in WORK)
                    blocks[key,r['alg']]=int(r['m'])
            # Per-process medians, then the median across processes.
            for key, algs in local.items():
                for alg, values in algs.items(): groups[key][alg].append(statistics.median(values))
    return [(fields[k],{a:statistics.median(v) for a,v in algs.items()},{a:works.get((k,a)) for a in algs},{a:blocks.get((k,a)) for a in algs}) for k,algs in groups.items()]

def rework(rows, paths):
    """Replace each timing's work vector by one from a --work-only run of the same workloads,
    so a changed model refits earlier timings: rect rows at the block length they ran."""
    work={}
    for pattern in paths:
        for path in glob.glob(pattern):
            with open(path) as f:
                for r in csv.DictReader(f):
                    work[tuple(r[k] for k in KEY),r['alg']]=tuple(float(r['w_'+w]) for w in WORK)
    missing=0
    for r,ts,ws,ms in rows:
        key=tuple(r[k] for k in KEY)
        for alg in ts:
            label={'Horner':'Horner','BinSplit':'BinSplit','Dispatch':r['picked']}.get(alg)
            if alg=='Rect' or alg.startswith('M'): label=f"W{ms[alg]}"
            if (key,label) in work: ws[alg]=work[key,label]
            else: missing+=1; ws[alg]=None
    if missing: print(f'{missing} timings without a reworked work vector are left out')

def constants(path):
    if path: return json.loads(pathlib.Path(path).read_text())['ns']
    text=(ROOT/'src/utils/mod.rs').read_text()
    return [float(re.search(rf'pub const {c}: f64 = ([-0-9.e]+);',text).group(1)) for c in CONSTS]

def predict(c,w): return sum(a*b for a,b in zip(c,w))

def geo(xs): return math.exp(statistics.mean(map(math.log,xs)))
def p95(xs): xs=sorted(xs); return xs[min(len(xs)-1,math.ceil(.95*len(xs))-1)]

def samples(rows):
    """Every timed algorithm with a work vector: forced Horner, Rect, BinSplit and rect blocks."""
    for r,ts,ws,_ in rows:
        for alg,ns in ts.items():
            if alg!='Dispatch' and ws.get(alg) is not None:
                yield r,('Rect' if alg.startswith('M') else alg),ws[alg],ns

def solve(a,b):
    """Least squares by normal equations (Gaussian elimination with pivoting); None if singular."""
    n=len(a[0]); m=[[sum(r[i]*r[j] for r in a) for j in range(n)]+[sum(r[i]*y for r,y in zip(a,b))] for i in range(n)]
    for i in range(n):
        p=max(range(i,n),key=lambda k:abs(m[k][i]))
        if abs(m[p][i])<1e-12: return None
        m[i],m[p]=m[p],m[i]
        for k in range(n):
            if k!=i:
                f=m[k][i]/m[i][i]
                m[k]=[x-f*y for x,y in zip(m[k],m[i])]
    return [m[i][n]/m[i][i] for i in range(n)]

def nnls(a,b):
    """Exact non-negative least squares for a few columns: the optimum is the unconstrained
    solution on its own support, so try every support and keep the best feasible one."""
    n=len(a[0]); norms=[math.sqrt(sum(r[j]**2 for r in a)) or 1. for j in range(n)]
    a=[[x/s for x,s in zip(r,norms)] for r in a]
    best=([0.]*n,sum(y*y for y in b))
    for k in range(1,n+1):
        for support in itertools.combinations(range(n),k):
            x=solve([[r[j] for j in support] for r in a],b)
            if x is None or min(x)<0: continue
            full=[0.]*n
            for j,v in zip(support,x): full[j]=v
            res=sum((predict(full,r)-y)**2 for r,y in zip(a,b))
            if res<best[1]-1e-15: best=(full,res)
    return [x/s for x,s in zip(best[0],norms)]

def errors(rows,c,intercept=0.):
    """Prediction error per mode/family/algorithm: typical factor, p95 factor and bias."""
    cells=collections.defaultdict(list)
    for r,alg,w,ns in samples(rows):
        cells[r['mode'],r['family'],alg].append(math.log((predict(c,w)+intercept)/ns))
    print(f'{"cell":24} {"n":>4} {"typical":>8} {"p95":>8} {"bias":>8}')
    for cell,ls in sorted(cells.items()):
        print(f'{"/".join(cell):24} {len(ls):4} {math.exp(statistics.mean(map(abs,ls))):8.3f} {math.exp(p95(list(map(abs,ls)))):8.3f} {math.exp(statistics.mean(ls)):8.3f}')
    every=[abs(l) for ls in cells.values() for l in ls]
    print(f'all: typical {math.exp(statistics.mean(every)):.3f}, p95 {math.exp(p95(every)):.3f}')

def regret(rows,c=None,verbose=True):
    """Balanced geometric regret against each workload's fastest forced algorithm, of the
    model's argmin over forced timings (c given) or of the measured actual dispatch."""
    groups=collections.defaultdict(list); worst=[]
    for r,ts,ws,_ in rows:
        forced=[a for a in ALGS if a in ts]
        if not forced: continue
        best=min(ts[a] for a in forced)
        if c is None:
            if 'Dispatch' not in ts: continue
            picked,elapsed=r['picked'],ts['Dispatch']
        else:
            if any(ws.get(a) is None for a in forced): continue
            picked=min(forced,key=lambda a:predict(c,ws[a])); elapsed=ts[picked]
        groups[r['mode']+'/'+r['family']].append(elapsed/best)
        worst.append((round(elapsed/best,3),r['mode'],r['family'],r['shape'],r['len'],picked,min(forced,key=ts.get)))
    if not groups: return None
    if verbose:
        for group,rs in sorted(groups.items()):
            print(f'{group:15} n={len(rs):3} geo={geo(rs):.4f} p95={p95(rs):.4f} max={max(rs):.4f}')
        print('Worst (regret, mode, family, shape, len, picked, fastest):',*sorted(worst,reverse=True)[:12],sep='\n')
    score=geo([geo(rs) for rs in groups.values()])
    if verbose: print(f'Balanced geometric regret: {score:.5f}')
    return score

def fit(args,rows):
    data=list(samples(rows))
    if not data: raise SystemExit('no rows with work columns')
    # Relative error: each row is (c . w + c0) / T against 1, with a free per-call intercept
    # c0 that every algorithm shares, so it cancels in dispatch and isn't shipped.
    a=[[x/ns for x in w]+[1/ns] for _,_,w,ns in data]
    sol=nnls(a,[1.]*len(a))
    c,c0=sol[:-1],sol[-1]
    print(f'{len(data)} timings; shared per-call intercept {c0:.1f} ns (not shipped)')
    for name,v in zip(CONSTS,c): print(f'pub const {name}: f64 = {v:.4g};')
    print('\nPrediction error (factor), with the intercept:'); errors(rows,c,c0)
    print('\nModel selection over these forced timings:'); regret(rows,c)
    if args.output: pathlib.Path(args.output).write_text(json.dumps({'ns':c,'intercept':c0},indent=2)+'\n')

def blocks(rows):
    """Production block length (forced Rect) against the best explicit one, M{m}."""
    out=collections.defaultdict(list)
    for r,ts,ws,ms in rows:
        explicit={a:v for a,v in ts.items() if a.startswith('M')}
        if 'Rect' not in ts or not explicit: continue
        best=min(explicit,key=explicit.get)
        out[r['mode']].append(ts['Rect']/explicit[best])
        print(f"{r['mode']:6} {r['shape']:9} len={r['len']:>4} t={r['t']:>5} xl={r['xl']:>3} model m={ms['Rect']:>4} best {best:>5} regret {ts['Rect']/explicit[best]:.3f}")
    for mode,rs in sorted(out.items()): print(f'{mode}: n={len(rs)} geo={geo(rs):.4f} max={max(rs):.4f}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    runp=sub.add_parser('run');runp.add_argument('--tag',required=True);runp.add_argument('--reps',type=int,default=3)
    for name in ('fit','report','blocks'):
        s=sub.add_parser(name)
        s.add_argument('paths',nargs='+')
        s.add_argument('--oracle',nargs='+',help='forced timings from another run for the same workloads')
        s.add_argument('--work',nargs='+',help='a --work-only run of the same workloads, for a changed model')
        if name=='fit': s.add_argument('--output')
        if name=='report': s.add_argument('--constants',help="fit's --output JSON; default: mod.rs")
    args,extra=p.parse_known_args()
    if args.cmd=='run': return run(args, extra[1:] if extra[:1]==['--'] else extra)
    if extra:p.error(str(extra))
    rows=load(args.paths)
    if not rows:raise SystemExit('no rows')
    if args.oracle:
        oracle={tuple(r[k] for k in KEY):(ts,ws) for r,ts,ws,_ in load(args.oracle)}
        for r,ts,ws,_ in rows:
            ots,ows=oracle.get(tuple(r[k] for k in KEY),({},{}))
            for alg in ALGS:
                if alg in ots: ts.setdefault(alg,ots[alg]); ws.setdefault(alg,ows.get(alg))
    if args.work: rework(rows,args.work)
    if args.cmd=='fit': fit(args,rows)
    elif args.cmd=='blocks': blocks(rows)
    else:
        print('Actual dispatch:')
        if regret(rows) is None: print('(no Dispatch rows)')
        if any(w is not None for _,_,ws,_ in rows for w in ws.values()):
            c=constants(args.constants)
            print('\nModel selection over forced timings:'); regret(rows,c)
            print('\nPrediction error (factor), without the intercept:'); errors(rows,c)
if __name__=='__main__':main()
