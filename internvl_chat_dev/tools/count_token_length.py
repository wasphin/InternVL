import json as json
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from transformers import AutoTokenizer

# set your path and output path here
# this code will find all `.jsonl` files and count the token length
path = '/mnt/petrelfs/wangwenhai/private_data/tianhao_data/v1_20240222_updateRoot_std'
output = '/mnt/petrelfs/wangwenhai/private_data/tianhao_data/v1_20240318_updateRoot_std'

model_path = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat/work_dirs/internvl_chat_internlm2_20b_448_dynamic_chinese_pretrain/checkpoint-5200'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

file_paths = []
# list all files in the directory and its subdirectories
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        # ignore temp directories and files
        if '_temp' not in file_path:
            file_paths.append(file_path)

min_num = 1
max_num = 6
target_ratios = set(
    (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
    i * j <= max_num and i * j >= min_num)
#  sort the ratios by their area
target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
print(target_ratios)
#  save middle result to speed up calculation
fast_dict= {}


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
        # remove rules made by hand
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def process_file(file_path):
    output_path = file_path.replace(path, output)
    # mkdir dir for output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open(output_path, 'w')
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            line = json.loads(line)
            if 'image' in line:
                width = line['width'] if 'width' in line else 448 * 2
                height = line['height'] if 'height' in line else 448 * 3
                aspect_ratio = width / height
                best_ratio = find_closest_aspect_ratio(
                    aspect_ratio, target_ratios, width, height, image_size=448)
                num_patch = best_ratio[0] * best_ratio[1]
                if num_patch == 1:
                    image_token = 256  # no thumbnail if only one patch
                    image_count = 1
                else:
                    image_token = (num_patch + 1) * 256  # add a thumbnail
                    image_count = num_patch + 1
            else:
                image_token = 0
                image_count = 0
            conversations = line['conversations']
            conversations = '\n'.join([item['value'] for item in conversations])
            tokenized = tokenizer(
                conversations, return_tensors='pt', padding=False, max_length=4096, truncation=True).input_ids
            text_token = tokenized.shape[1]
            line['length'] = image_token + text_token
            line['image_count'] = image_count
            writer.write(json.dumps(line, ensure_ascii=False) + '\n')
    writer.close()
    print('finished!')


# 使用ProcessPoolExecutor并行处理文件
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
