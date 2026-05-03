import csv
import json
import os
from pathlib import Path

"""
Simple CSV -> disease_info.json importer.

CSV columns (header row required):
  class_name,common_name,pesticide_recommendation,organic_alternatives,care_tips,source,safety_notes

Usage:
  python tools/import_disease_info.py data/disease_info.csv

If output file exists it will be backed up to data/disease_info.json.bak
"""


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def build_dict(rows):
    out = {}
    for r in rows:
        cls = r.get('class_name') or r.get('class')
        if not cls:
            print('Skipping row with no class_name:', r)
            continue
        out[cls] = {
            'common_name': r.get('common_name', '').strip(),
            'pesticide_recommendation': r.get('pesticide_recommendation', '').strip(),
            'organic_alternatives': r.get('organic_alternatives', '').strip(),
            'care_tips': r.get('care_tips', '').strip(),
            'source': r.get('source', '').strip(),
            'safety_notes': r.get('safety_notes', '').strip()
        }
    return out


def write_json(obj, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + '.bak')
        print(f'Backing up existing {out_path} -> {bak}')
        out_path.rename(bak)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f'Wrote {out_path} ({len(obj)} entries)')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python tools/import_disease_info.py path/to/disease_info.csv [out_json_path]')
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print('CSV not found:', csv_path)
        sys.exit(1)
    rows = read_csv(csv_path)
    data = build_dict(rows)
    # optional second argument: output JSON path (useful for domain-specific files)
    if len(sys.argv) >= 3:
        out = Path(sys.argv[2])
    else:
        out = Path('data/disease_info.json')
    os.makedirs(out.parent, exist_ok=True)
    write_json(data, out)
