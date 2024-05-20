import json
import logging
import os
import random
import warnings
from copy import deepcopy

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from internvl.train.dataset import (build_transform, preprocess,
                                    preprocess_internlm, preprocess_llama3,
                                    preprocess_mpt, preprocess_phi3)
from PIL import ImageFile, PngImagePlugin
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

import io

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from typing import Dict

import torch
import torchvision.transforms as T
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode


class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn):
        img_value_str = self.client.get(fn)
        img = pil_loader(img_value_str)
        return img


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def build_transform(is_train, input_size, pad2square=False):
    if is_train:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.),
                                interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        if pad2square is False:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    return transform


IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

f = open('shell/data/debug.json')
data = json.load(f)
ds_collections = {}
for k, v in data.items():
    print(k, v)
    ds_collections[k] = {
        'root': v['root'],
        'annotation': v['annotation'],
        'data_augment': v['data_augment'],
        'repeat_time': v['repeat_time'],
        'length': v['length'],
    }


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, tcs_loader, num_image_token,
                 image_size=224, is_train=True, pad2square=False, group_by_length=False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.group_by_length = group_by_length
        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader

    def __len__(self):
        return len(self.raw_data)

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
        if best_ratio == (2, 3) or best_ratio == (3, 2):
            new_area = self.image_size * self.image_size * 4
            if area < new_area:
                best_ratio = (2, 2)
        if best_ratio == (1, 1) or best_ratio == (2, 2):
            if area < self.image_size * self.image_size:
                best_ratio = (1, 1)
            else:
                best_ratio = (2, 2)
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=6):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # 计算理想的块数
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)

        # 根据理想的块数找到最接近的目标长宽比和块数
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height)

        # 计算目标尺寸
        target_width = self.image_size * target_aspect_ratio[0]
        target_height = self.image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # 调整图像大小和分割
        resized_img = image.resize((target_width, target_height))
        # 返回处理后的图像列表
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.image_size)) * self.image_size,
                (i // (target_width // self.image_size)) * self.image_size,
                ((i % (target_width // self.image_size)) + 1) * self.image_size,
                ((i // (target_width // self.image_size)) + 1) * self.image_size
            )
            split_img = resized_img.crop(box)
            split_img.save(f'{i}.png')
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        return processed_images

    def multi_modal_get_item(self, data_item):
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        if data_item['image'].startswith('s3://'):
            image_path = self.root + data_item['image']
        else:
            image_path = os.path.join(self.root, data_item['image'])
        image = self.tcs_loader(image_path)
        images = self.dynamic_preprocess(image)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'llama3-chat':
            preprocess_function = preprocess_llama3
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * num_patches,
                                  group_by_length=self.group_by_length)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        images = self.dynamic_preprocess(image)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'llama3-chat':
            preprocess_function = preprocess_llama3
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * num_patches,
                                  text_only=True, group_by_length=self.group_by_length)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and len(data_item['image']) != 0:
                    ret = self.multi_modal_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                logger.info(e)
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if data_item['image'].startswith('s3://'):
                        data_path = self.root + data_item['image']
                    else:
                        data_path = os.path.join(self.root, data_item['image'])
                    print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def test_dataset(dataset):
    utilization_sum = 0
    for _ in tqdm(range(10)):
        index = random.randint(0, len(dataset) - 1)
        data = dataset.__getitem__(index)
        input_ids = data['input_ids']
        utilization = 1 - ((input_ids == llm_tokenizer.pad_token_id).sum() / input_ids.numel())
        utilization_sum += utilization
    print('utilization:', utilization_sum / 100 * llm_tokenizer.model_max_length)
    print('utilization rate:', utilization_sum / 100)


if __name__ == '__main__':
    llm_path = './pretrained/Meta-Llama-3-8B-Add-Token'
    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_path, add_eos_token=False, trust_remote_code=True)
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN,
                  '<|system|>', '<|user|>', '<|assistant|>', '<|end|>']
    num_new_tokens = llm_tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = llm_tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    llm_tokenizer.model_max_length = 3072
    tcs_loader = TCSLoader('~/petreloss.conf')

    for ds_name in ds_collections.keys():
        print(f'Testing {ds_name}...')
        dataset = LazySupervisedDataset(
            'llama3-chat',
            ds_collections[ds_name],
            llm_tokenizer,
            tcs_loader,
            num_image_token=256,
            image_size=448,
            is_train=ds_collections[ds_name]['data_augment'],
        )
        dataset.ds_name = ds_name
        test_dataset(dataset)
