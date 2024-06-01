import json as json
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

path = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/metas/stage3_v5_20240521_std/internlm/'
output = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/metas/stage3_v5_20240521_std/internlm_clean/'


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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open(output_path, 'w')

    count = 0
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            line = json.loads(line)
            conversations = line['conversations']
            print_flag = True
            for conv in conversations:
                if conv['value'] is None or len(conv['value']) == 0:
                    print_flag = False
            if print_flag:
                writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()

    if count > 0:
        print(file_path, count, len(data))


# 使用ProcessPoolExecutor并行处理文件
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
