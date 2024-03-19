import json as json
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

path = '/mnt/petrelfs/wangwenhai/private_data/tianhao_data/v1_20240318_updateRoot_std'
output = '/mnt/petrelfs/wangwenhai/private_data/tianhao_data/v1_20240318_updateRoot_std_filtered3072'
allow_max_length = 3072

file_paths = []
# list all files in the directory and its subdirectories
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        if '_temp' not in file_path:
            file_paths.append(file_path)


def process_file(file_path):
    basename = os.path.basename(file_path)
    output_path = file_path.replace(path, output)
    # mkdir dir for output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open(output_path, 'w')
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            item = json.loads(line)
            length = item['length']
            if length <= allow_max_length:
                writer.write(line)
    writer.close()
    print('finished!')


# 使用ProcessPoolExecutor并行处理文件
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
