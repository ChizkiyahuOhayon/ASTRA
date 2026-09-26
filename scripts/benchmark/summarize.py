"""Print a Markdown table of every finished run under an output root.

    python scripts/benchmark/summarize.py output/waymo/astra

One row per scene, then the mean. KITTI runs (`<seq>-nvs<split>`) are averaged per
split over the three sequences, which is how AD-GS reports KITTI-MOT.
"""
import json
import os
import re
import sys
from collections import defaultdict

COLUMNS = [('PSNR', 'PSNR'), ('SSIM', 'SSIM'), ('LPIPS(VGG)', 'LPIPS-VGG'), ('LPIPS(ALEX)', 'LPIPS-Alex')]


def load(run):
    with open(os.path.join(run, 'results.json')) as handle:
        metrics = next(iter(json.load(handle).values()))
    row = {name: metrics[key] for key, name in COLUMNS}
    dpsnr = os.path.join(run, 'results-dpsnr.json')
    if os.path.exists(dpsnr):
        with open(dpsnr) as handle:
            row['DPSNR'] = json.load(handle)['DPSNR']
    return row


def table(rows):
    names = [name for _, name in COLUMNS] + (['DPSNR'] if all('DPSNR' in r for r in rows.values()) else [])
    lines = ['| | ' + ' | '.join(names) + ' |', '|---' * (len(names) + 1) + '|']
    for label, row in rows.items():
        lines.append('| %s | ' % label + ' | '.join('%.4f' % row[n] for n in names) + ' |')
    return '\n'.join(lines)


def mean(rows):
    keys = set.intersection(*(set(r) for r in rows))
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


if __name__ == '__main__':
    root = sys.argv[1]
    runs = {d: load(os.path.join(root, d)) for d in sorted(os.listdir(root))
            if os.path.exists(os.path.join(root, d, 'results.json'))}
    if not runs:
        sys.exit('no finished runs under ' + root)
    splits = defaultdict(list)
    for name, row in runs.items():
        match = re.search(r'-nvs(\d+)$', name)
        if match:
            splits['nvs-' + match.group(1)].append(row)
    print(table(runs))
    print()
    if splits:
        print(table({'%s mean (%d seq)' % (s, len(r)): mean(r) for s, r in sorted(splits.items(), reverse=True)}))
    else:
        print(table({'mean (%d scenes)' % len(runs): mean(list(runs.values()))}))
