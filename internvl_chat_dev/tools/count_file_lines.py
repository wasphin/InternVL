import argparse
import json

argparse = argparse.ArgumentParser()
argparse.add_argument('-n', '--name', type=str, required=True)
args = argparse.parse_args()

print(f'Hello, {args.name}!')
data = json.loads(open(args.name).read())
for k, v in data.items():
    annotation = v['annotation']
    length = v['length']
    f = open(annotation, 'r')
    lines = f.readlines()
    f.close()
    if len(lines) != length:
        print(f'{k}: {annotation} has {len(lines)} lines, expected {length}')
    v['length'] = len(lines)
json.dump(data, open(args.name, 'w'), indent=2, ensure_ascii=False)
