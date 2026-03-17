#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_node_metadata(metadata_glob: str):
    node_info = {}
    for p in sorted(Path('.').glob(metadata_glob)):
        with p.open('r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row['node_id'])
                node_info[node_id] = {
                    'node_name': row.get('node_name', ''),
                    'node_type': row.get('node_type', ''),
                    'comm_size': int(row.get('comm_size', '0') or 0),
                }
    return node_info


def _build_tag_to_node_map(events_csv: Path, node_info: dict) -> dict:
    """
    For collective comms, node_id in events is always 0 (the collective
    machinery doesn't propagate the ET node_id through the ring steps).
    Instead, each ET node generates exactly `preferred_splits` consecutive tags
    in sequential execution order (since layers are data_dep chained).

    This function maps each unique tag → ET node_id by:
      1. Sorting unique tags numerically.
      2. Dividing into len(node_info) equal groups of tags.
      3. Assigning group[i] → ET node sorted by node_id at rank i.

    Falls back to identity mapping (tag → 0) if node_info has only one entry.
    """
    if len(node_info) <= 1:
        return {}  # nothing to do

    unique_tags: list[int] = []
    with events_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        seen = set()
        for row in reader:
            t = int(row['comm_tag'])
            if t not in seen:
                seen.add(t)
                unique_tags.append(t)
    unique_tags.sort()

    n_layers = len(node_info)
    tags_per_layer, remainder = divmod(len(unique_tags), n_layers)
    if remainder != 0 or tags_per_layer == 0:
        # Cannot partition evenly — fall back to raw node_id
        return {}

    ordered_node_ids = sorted(node_info.keys())
    tag_to_node: dict[int, int] = {}
    for layer_rank, et_node_id in enumerate(ordered_node_ids):
        start = layer_rank * tags_per_layer
        for tag in unique_tags[start: start + tags_per_layer]:
            tag_to_node[tag] = et_node_id
    return tag_to_node


def build_profile(events_csv: Path, node_info: dict):
    # Build tag→node_id map for collective comm traces where node_id=0 always
    tag_to_node = _build_tag_to_node_map(events_csv, node_info)
    use_tag_map = bool(tag_to_node)

    acc = defaultdict(lambda: {
        'events': 0,
        'quantized_events': 0,
        'original_bytes': 0,
        'effective_bytes': 0,
        'comm_src_set': set(),
        'comm_dst_set': set(),
    })

    with events_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if use_tag_map:
                tag = int(row['comm_tag'])
                node_id = tag_to_node.get(tag, int(row['node_id']))
            else:
                node_id = int(row['node_id'])
            item = acc[node_id]
            item['events'] += 1
            item['quantized_events'] += int(row['quantized'])
            item['original_bytes'] += int(row['original_bytes'])
            item['effective_bytes'] += int(row['effective_bytes'])
            item['comm_src_set'].add(int(row['comm_src']))
            item['comm_dst_set'].add(int(row['comm_dst']))

    profile = []
    for node_id, item in sorted(acc.items(), key=lambda x: x[0]):
        quantized_event_ratio = (
            item['quantized_events'] / item['events'] if item['events'] else 0.0
        )
        byte_reduction_ratio = (
            1.0 - (item['effective_bytes'] / item['original_bytes'])
            if item['original_bytes'] else 0.0
        )
        meta = node_info.get(node_id, {})
        profile.append({
            'node_id': node_id,
            'node_name': meta.get('node_name', ''),
            'node_type': meta.get('node_type', ''),
            'comm_size': meta.get('comm_size', 0),
            'events': item['events'],
            'quantized_events': item['quantized_events'],
            'quantized_event_ratio': quantized_event_ratio,
            'original_bytes': item['original_bytes'],
            'effective_bytes': item['effective_bytes'],
            'byte_reduction_ratio': byte_reduction_ratio,
            'comm_src_count': len(item['comm_src_set']),
            'comm_dst_count': len(item['comm_dst_set']),
        })
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Build per-layer quantization profile from phase-3 event logs'
    )
    parser.add_argument('--events-csv', required=True, help='Path to quantization_events.csv')
    parser.add_argument(
        '--metadata-glob',
        default='results/phase3/comm_node_metadata_rank*.csv',
        help='Glob for comm node metadata CSVs',
    )
    parser.add_argument(
        '--output-json',
        default='results/phase4/layer_quant_profile.json',
        help='Output JSON profile',
    )
    parser.add_argument(
        '--output-csv',
        default='results/phase4/layer_quant_profile.csv',
        help='Output CSV profile',
    )
    args = parser.parse_args()

    events_csv = Path(args.events_csv)
    if not events_csv.exists():
        raise FileNotFoundError(f'events csv not found: {events_csv}')

    node_info = load_node_metadata(args.metadata_glob)
    tag_map = _build_tag_to_node_map(events_csv, node_info)
    if tag_map:
        print(f"  [profile] Using tag-based layer assignment ({len(node_info)} layers, "
              f"{len(set(tag_map.values()))} mapped)")
    else:
        print(f"  [profile] Using node_id-based layer assignment ({len(node_info)} layers)")
    profile = build_profile(events_csv, node_info)

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open('w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)

    with out_csv.open('w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'node_id',
            'node_name',
            'node_type',
            'comm_size',
            'events',
            'quantized_events',
            'quantized_event_ratio',
            'original_bytes',
            'effective_bytes',
            'byte_reduction_ratio',
            'comm_src_count',
            'comm_dst_count',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in profile:
            writer.writerow(row)

    print('=== Phase 4 Layer Quantization Profile ===')
    print(f'events_csv={events_csv}')
    print(f'metadata_glob={args.metadata_glob}')
    print(f'output_json={out_json}')
    print(f'output_csv={out_csv}')
    print(f'layers={len(profile)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
