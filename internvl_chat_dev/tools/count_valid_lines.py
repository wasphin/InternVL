import argparse
import json

argparse = argparse.ArgumentParser()
argparse.add_argument('-n', '--name', type=str, required=True)
args = argparse.parse_args()

print(f'Hello, {args.name}!')
data = json.loads(open(args.name).read())
sum = 0
for k, v in data.items():
    annotation = v['annotation']
    length = v['length']
    repeat_time = v['repeat_time']
    sum += length * repeat_time
print(sum)
