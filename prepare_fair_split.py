#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--train', type=int, default=200)
    ap.add_argument('--eval', type=int, default=300)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw_by_kind = {
        kind: json.loads((args.source / f'{kind}.json').read_text())
        for kind in ('pointwise', 'pairwise', 'listwise')
    }
    ids = [str(r.get('id')) for r in raw_by_kind['pointwise']]
    if len(set(ids)) != len(ids):
        raise ValueError('pointwise ids are not unique')
    by_kind = {kind: {str(r.get('id')): r for r in records} for kind, records in raw_by_kind.items()}
    for kind in ('pairwise', 'listwise'):
        if set(by_kind[kind]) != set(ids):
            raise ValueError(f'{kind} ids do not match pointwise ids')
    order = list(ids)
    if args.shuffle:
        random.Random(args.seed).shuffle(order)
    if len(order) < args.offset + args.train + args.eval:
        raise ValueError(f'source has {len(order)} records, need {args.offset + args.train + args.eval}')
    selected_ids = order[args.offset:args.offset + args.train + args.eval]
    for kind in ('pointwise', 'pairwise', 'listwise'):
        records = [by_kind[kind][record_id] for record_id in selected_ids]
        train_records = records[:args.train]
        eval_records = records[args.train:]
        (args.out / f'{kind}_train{args.train}.json').write_text(
            json.dumps(train_records, ensure_ascii=False, indent=2) + '\n')
        (args.out / f'{kind}_eval{args.eval}.json').write_text(
            json.dumps(eval_records, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
