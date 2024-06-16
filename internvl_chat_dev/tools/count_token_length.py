import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from internvl.train.dataset import TCSLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Initialize the dataset loader
tcs_loader = TCSLoader('~/petreloss.conf')

# Set your path and output path here
# This code will find all `.jsonl` files and count the token length
path = '/mnt/petrelfs/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/metas/stage3_v5_20240611_std/gpt4o/temp'
output = '/mnt/petrelfs/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/metas/stage3_v5_20240611_std/gpt4o/temp2'

# Model path for the tokenizer
model_path = '/mnt/hwfile/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/work_dirs/internvl_chat_v1_5/' \
             'internvl_chat_v1_5_internlm2_20b_dynamic_res_finetune_exp7_26'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# Load metadata
meta_path = './shell/data/data_yi34b_finetune_v5_45.json'
meta = json.load(open(meta_path, 'r'))
basename2meta = {os.path.basename(v['annotation']): v for k, v in meta.items()}

# Collect all file paths
file_paths = []
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        if '_temp' not in file_path:
            file_paths.append(file_path)
file_paths = [f for f in file_paths if f.endswith('.jsonl')]
print(file_paths)


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
    return best_ratio


def process_image(line, root):
    if 'image' in line:
        if isinstance(line['image'], list):
            if len(line['image']) == 1:
                line['image'] = line['image'][0]
                return process_single_image(line, root)
            else:
                return process_image_list(line, root)
        else:
            return process_single_image(line, root)
    return line


def process_image_list(line, root):
    if 'width_list' not in line or 'height_list' not in line:
        width_list, height_list, image_list = [], [], []
        for url in line['image']:
            image_path = os.path.join(root, url)
            try:
                image = tcs_loader(image_path)
                width, height = image.size
                width_list.append(width)
                height_list.append(height)
                image_list.append(url)
            except:
                continue
        if not image_list:
            return None
        line['image'] = image_list
        line['width_list'] = width_list
        line['height_list'] = height_list
        line['width'] = width_list[0]
        line['height'] = height_list[0]
    return line


def process_single_image(line, root):
    image_path = os.path.join(root, line['image'])
    if 'width' not in line or 'height' not in line:
        try:
            image = tcs_loader(image_path)
            width, height = image.size
            line['width'] = width
            line['height'] = height
            if 'width_list' in line:
                del line['width_list']
            if 'height_list' in line:
                del line['height_list']
        except:
            pass
    return line


def calculate_image_tokens(line, image_size=448):
    width_list = line.get('width_list', [line['width']])
    height_list = line.get('height_list', [line['height']])
    num_patch_total = 0
    num_image = len(width_list)
    for width, height in zip(width_list, height_list):
        aspect_ratio = width / height
        min_num, max_num = 1, 12 // num_image
        target_ratios = {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num}
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        best_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size=image_size)
        num_patch = best_ratio[0] * best_ratio[1]
        if num_patch != 1:
            num_patch += 1  # Add a thumbnail
        num_patch_total += num_patch
    image_token = num_patch_total * 256
    image_count = num_patch_total
    return image_token, image_count


def tokenize_conversations(line):
    try:
        conversations = line['conversations']
    except KeyError:
        raise ValueError('Conversations key missing in the line')
    conversations = '\n'.join([item['value'] for item in conversations])
    tokenized = tokenizer(conversations, return_tensors='pt', padding=False, max_length=99999, truncation=True).input_ids
    text_token = tokenized.shape[1]
    return text_token


def process_file(file_path):
    basename = os.path.basename(file_path)
    root = basename2meta.get(basename, {}).get('root', '')

    output_path = file_path.replace(path, output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open(output_path, 'w')

    diff_length = []
    try:
        with open(file_path, 'r') as f:
            data = f.readlines()
            for line in tqdm(data):
                try:
                    line = json.loads(line)
                except:
                    continue

                line = process_image(line, root)
                if line is None:
                    continue

                if 'image' in line:
                    image_token, image_count = calculate_image_tokens(line)
                else:
                    image_token, image_count = 0, 0

                text_token = tokenize_conversations(line)

                diff_length.append(image_token + text_token - line.get('length', 0))
                line['length'] = image_token + text_token
                line['image_count'] = image_count
                writer.write(json.dumps(line, ensure_ascii=False) + '\n')
        writer.close()
        diff_length = np.array(diff_length).mean()
        print(f'{basename} finished! diff: {diff_length}')
    except Exception as e:
        import traceback
        traceback.print_exc()


# Use ProcessPoolExecutor to process files in parallel
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
