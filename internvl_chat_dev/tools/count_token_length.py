import json as json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from internvl.train.dataset import TCSLoader
from tqdm import tqdm
from transformers import AutoTokenizer

tcs_loader = TCSLoader('~/petreloss.conf')
try_to_load_image = True

# set your path and output path here
# this code will find all `.jsonl` files and count the token length
path = '/mnt/hwfile/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/metas/medical_data/merged_share_datasets/'
output = '/mnt/hwfile/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/metas/stage3_v5_20240611_std/medical/'

model_path = '/mnt/hwfile/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/work_dirs/internvl_chat_v1_5/' \
             'internvl_chat_v1_5_internlm2_20b_dynamic_res_finetune_exp7_26'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

meta_path = './shell/data/data_yi34b_finetune_v5_35.json'
meta = json.load(open(meta_path, 'r'))
basename2meta = {}
for k, v in meta.items():
    annotation = v['annotation']
    basename = os.path.basename(annotation)
    basename2meta[basename] = v

file_paths = []
# list all files in the directory and its subdirectories
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        # ignore temp directories and files
        if '_temp' not in file_path:
            file_paths.append(file_path)
file_paths = [f for f in file_paths if f.endswith('.jsonl')]


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


def process_file(file_path):
    basename = os.path.basename(file_path)
    if basename in basename2meta:
        meta = basename2meta[basename]
        root = meta['root']
    else:
        root = ''

    output_path = file_path.replace(path, output)
    # mkdir dir for output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = open(output_path, 'w')

    diff_length = []
    min_num, max_num = 1, 12
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    try:
        with open(file_path, 'r') as f:
            data = f.readlines()
            for line in tqdm(data):
                try:
                    line = json.loads(line)
                except:
                    # 解析失败直接跳过
                    continue
                # 删掉多余的image_root
                if 'image_root' in line:
                    del line['image_root']
                # image只有多余1条时才能用list
                if 'image' in line and type(line['image']) == list and len(line['image']) == 1:
                    line['image'] = line['image'][0]
                if 'image' in line and type(line['image']) == str:
                    line['image'] = line['image'].replace('s3://medical_preprocessed/', '')

                if 'image' in line:
                    if type(line['image']) == list:
                        pass
                    else:
                        # 如果是单图的数据，并且长宽未知时，去读取数据获取长宽
                        image_path = os.path.join(root, line['image'])
                        if 'width' not in line or 'height' not in line:
                            try:
                                image = tcs_loader(image_path)
                                width, height = image.size
                                line['width'] = width
                                line['height'] = height
                            except:
                                # 读取失败时跳过
                                pass

                    # 多图先直接按12算
                    width = line['width'] if 'width' in line else 448 * 3
                    height = line['height'] if 'height' in line else 448 * 4
                    aspect_ratio = width / height
                    best_ratio = find_closest_aspect_ratio(
                        aspect_ratio, target_ratios, width, height, image_size=448)
                    num_patch = best_ratio[0] * best_ratio[1]

                    if num_patch == 1:
                        image_token = 256  # 只有1个块时，没有缩略图
                        image_count = 1
                    else:
                        if type(line['image']) == list:
                            image_count = len(line['image'])
                            image_token = image_count * 256
                            # 如果有多图，每个都是448x448
                        else:
                            image_token = (num_patch + 1) * 256  # add a thumbnail
                            image_count = num_patch + 1
                            # 否则按动态分辨率统计
                else:
                    image_token = 0
                    image_count = 0
                try:
                    conversations = line['conversations']
                except:
                    print(file_path)
                    exit()
                conversations = '\n'.join([item['value'] for item in conversations])
                tokenized = tokenizer(
                    conversations, return_tensors='pt', padding=False, max_length=99999, truncation=True).input_ids
                text_token = tokenized.shape[1]
                if 'length' in line:
                    diff_length.append(image_token + text_token - line['length'])
                else:
                    diff_length.append(0)
                line['length'] = image_token + text_token
                line['image_count'] = image_count
                writer.write(json.dumps(line, ensure_ascii=False) + '\n')
        writer.close()
        diff_length = np.array(diff_length).mean()
        print(f'{basename} finished! diff: {diff_length}')
    except:
        import traceback
        traceback.print_exc()


# 使用ProcessPoolExecutor并行处理文件
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
