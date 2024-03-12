import json as json
import os
from concurrent.futures import ProcessPoolExecutor

from internvl.train.dataset import find_closest_aspect_ratio
from tqdm import tqdm
from transformers import AutoTokenizer

fast_mode = False
path = '/mnt/petrelfs/wangwenhai/private_data/tianhao_data/v1_20240222_updateRoot_std'
output = '/mnt/petrelfs/wangwenhai/private_data/tianhao_data/v1_20240222_updateRoot_std2'
model_path = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat/work_dirs/internvl_chat_internlm2_20b_448_dynamic_chinese_pretrain/checkpoint-5200'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

file_paths = []
# list all files in the directory and its subdirectories
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        if '_temp' not in file_path:
            file_paths.append(file_path)

min_num = 1
max_num = 6
target_ratios = set(
    (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
    i * j <= max_num and i * j >= min_num)
print(target_ratios)
fast_dict= {}


def process_file(file_path):
    basename = os.path.basename(file_path)
    output_path = file_path.replace(path, output)
    # mkdir dir for output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open(output_path, 'w')
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            line = json.loads(line)
            if 'image' in line:
                if not fast_mode:
                    width = line['width'] if 'width' in line else 448 * 2
                    height = line['height'] if 'height' in line else 448 * 3
                    line['width'] = width
                    line['height'] = height
                    aspect_ratio = width / height
                    best_ratio = find_closest_aspect_ratio(
                        aspect_ratio, target_ratios, width, height, image_size=448)
                    sum_ratio = sum(best_ratio)
                    if sum_ratio == 1:
                        image_token = 256
                        image_count = 1
                    else:
                        image_token = sum_ratio * 256 + 256
                        image_count = sum_ratio + 1
                else:
                    image_token = 6 * 256 + 256
                    image_count = 7
            else:
                image_token = 0
                image_count = 1
            conversations = line['conversations']
            conversations = '\n'.join([item['value'] for item in conversations])
            if fast_mode:
                if len(conversations) not in fast_dict:
                    tokenized = tokenizer(
                        conversations, return_tensors='pt', padding=False, max_length=4096, truncation=True).input_ids
                    text_token = tokenized.shape[1]
                    length = image_token + text_token
                    line['length'] = length
                    fast_dict[len(conversations)] = length
                else:
                    line['length'] = fast_dict[len(conversations)]
            else:
                tokenized = tokenizer(
                    conversations, return_tensors='pt', padding=False, max_length=4096, truncation=True).input_ids
                text_token = tokenized.shape[1]
                length = image_token + text_token
                line['length'] = length
                line['image_count'] = image_count

            writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()
    print('finished!')


# 使用ProcessPoolExecutor并行处理文件
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
