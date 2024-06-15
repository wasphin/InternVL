
import argparse
import json

argparse = argparse.ArgumentParser()
argparse.add_argument('path', type=str)

args = argparse.parse_args()

assert args.path.endswith('.json')

data = json.loads(open(args.path).read())
writer = open(args.path.replace('.json', '.jsonl'), 'w')
for idx, item in enumerate(data):
    conversations = item['conversations']
    try:
        if conversations[0]['from'] == 'system':
            item['conversations'] = item['conversations'][1:]
    except:
        continue
    item['id'] = idx
    writer.write(json.dumps(item, ensure_ascii=False) + '\n')

writer.close()
