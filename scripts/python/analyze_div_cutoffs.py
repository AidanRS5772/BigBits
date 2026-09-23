#!/usr/bin/env python3
"""Fit division cost models or summarize paired runtime measurements.

Usage: python3 scripts/python/analyze_div_cutoffs.py docs/division_dispatch_grid.csv
The objective gives each operand shape and operation equal weight. Costs are
normalized to NR at each shape, so comparing shared cutoffs assumes NR timing
is stable between builds. Validate the result with production runtime benches.
By default the fit reproduces the historical shared-cutoff strategy. Supply
--knuth-cutoff to fit separate eligibility and recursive cutoffs; NR/Knuth
measurements are required wherever either cutoff can select Knuth.
"""

import argparse
import csv
import math


def feature(point, transform):
    q, d, _ = point
    return math.log2(d) ** 2 / math.log2(q) if transform else d / q


def fit_ratio(points, transform):
    values = sorted({feature(p, transform) for p in points})
    candidates = [values[0] * 0.9]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates += [values[-1] * 1.1]

    def loss(ratio):
        return sum(-math.log(p[2]) for p in points if feature(p, transform) < ratio)

    ratio = min(candidates, key=loss)
    return loss(ratio), ratio


def fit(rows, cutoff, knuth_cutoff=None):
    knuth_cutoff = cutoff if knuth_cutoff is None else knuth_cutoff
    fixed = sum(-math.log(float(r['nr_over_knuth'])) for r in rows if int(r['d']) <= knuth_cutoff)
    points = [(int(r['q']), int(r['d']),
               float(r['nr_over_knuth' if int(r['d']) <= cutoff else 'nr_over_bz']))
              for r in rows if int(r['d']) > knuth_cutoff]
    widths = sorted({max(p[:2]) for p in points})
    switches = [(a + b + 1) // 2 for a, b in zip(widths, widths[1:])]
    results = []
    for switch in switches:
        small = [p for p in points if max(p[:2]) < switch]
        large = [p for p in points if max(p[:2]) >= switch]
        # Avoid fitting a separate regime to just a few tail measurements.
        if min(len(small), len(large)) < 6:
            continue
        if min(len({p[1] for p in small}), len({p[1] for p in large})) < 2:
            continue
        a, karatsuba = fit_ratio(small, False)
        b, transform = fit_ratio(large, True)
        results.append((fixed + a + b, switch, karatsuba, transform))
    return min(results)


def model_loss(rows, cutoff, switch, karatsuba, transform, knuth_cutoff=None):
    knuth_cutoff = cutoff if knuth_cutoff is None else knuth_cutoff
    total = 0.0
    for row in rows:
        q, d = int(row['q']), int(row['d'])
        if d <= knuth_cutoff:
            total -= math.log(float(row['nr_over_knuth']))
        else:
            use_transform = max(q, d) >= switch
            ratio = transform if use_transform else karatsuba
            if feature((q, d, None), use_transform) < ratio:
                backend = 'nr_over_knuth' if d <= cutoff else 'nr_over_bz'
                total -= math.log(float(row[backend]))
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--knuth-cutoff', type=int, help='separate mandatory Knuth cutoff for model fitting')
    parser.add_argument('--runtime', action='store_true', help='summarize a runtime comparison CSV')
    parser.add_argument('--family', choices=['dyn_div', 'dyn_rem'])
    parser.add_argument('--model', nargs=4, type=float,
                        metavar=('BZ_CUTOFF', 'SWITCH', 'KARATSUBA', 'TRANSFORM'))
    args = parser.parse_args()
    if args.model and not args.family:
        parser.error('--model requires --family')
    with open(args.csv, newline='') as source:
        rows = list(csv.DictReader(source))
    if args.runtime:
        for family in ([args.family] if args.family else ['dyn_div', 'dyn_rem']):
            corpora = sorted({r['corpus'] for r in rows if r['family'] == family})
            for corpus in corpora + ['all']:
                selected = [r for r in rows if r['family'] == family
                            and (corpus == 'all' or r['corpus'] == corpus)]
                if not selected:
                    continue
                ratios = [float(r['after_ns']) / float(r['before_ns']) for r in selected]
                cost = math.exp(sum(map(math.log, ratios)) / len(ratios))
                print(f'{family} {corpus}: n={len(ratios)} geometric runtime={cost:.6f} '
                      f'range={min(ratios):.6f}..{max(ratios):.6f}')
        return
    cutoffs = sorted({int(r['bz_cutoff']) for r in rows})
    families = [args.family] if args.family else ['dyn_div', 'dyn_rem']
    joint = {cutoff: [0.0, 0] for cutoff in cutoffs}
    for family in families:
        for cutoff in cutoffs:
            selected = [r for r in rows if r['family'] == family and int(r['bz_cutoff']) == cutoff]
            loss, switch, karatsuba, transform = fit(selected, cutoff, args.knuth_cutoff)
            joint[cutoff][0] += loss
            joint[cutoff][1] += len(selected)
            print(f'{family}: BZ={cutoff} switch={switch} karatsuba={karatsuba:.6f} '
                  f'transform={transform:.6f} cost/NR={math.exp(loss / len(selected)):.6f}')
    if args.model:
        cutoff, switch, karatsuba, transform = args.model
        selected = [r for r in rows if r['family'] == args.family and int(r['bz_cutoff']) == cutoff]
        if not selected:
            parser.error('no measurements for the requested shared cutoff')
        loss = model_loss(selected, cutoff, switch, karatsuba, transform, args.knuth_cutoff)
        print(f'Specified model: cost/NR={math.exp(loss / len(selected)):.6f}')
    elif not args.family:
        for cutoff, (loss, count) in joint.items():
            print(f'Joint BZ={cutoff}: cost/NR={math.exp(loss / count):.6f}')


if __name__ == '__main__':
    main()
