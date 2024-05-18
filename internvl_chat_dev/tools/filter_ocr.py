import json as json
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# set your path and output path here
# this code will find all `.jsonl` files and count the token length
path = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/metas/stage3_v5_20240517_std/ocr'
output = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/metas/stage3_v5_20240517_std/ocr3'

file_paths = []
# list all files in the directory and its subdirectories
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        # ignore temp directories and files
        if '_temp' not in file_path:
            file_paths.append(file_path)


def process_file(file_path):
    output_path = file_path.replace(path, output)
    # mkdir dir for output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open(output_path, 'w')
    cnt = 0
    with open(file_path, 'r') as f:
        data = f.readlines()
        length = len(data)
        for line in tqdm(data):
            line = json.loads(line)
            conversations = line['conversations']
            conversations = '\n'.join([item['value'] for item in conversations if item['from'] == 'gpt'])
            conversations = conversations.split('\n')
            diff = len(conversations) - len(set(conversations))
            if diff >= 9:
                cnt += 1
                writer.write(json.dumps(line, ensure_ascii=False) + '\n')
                continue
            else:
                continue
    writer.close()
    print(f'{file_path} processed, {cnt} files are ignored, percentage: {cnt/length *  100:.2f}%')


# 使用ProcessPoolExecutor并行处理文件
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
