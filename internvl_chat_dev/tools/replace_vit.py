import argparse

import torch
from internvl.model.internvl_chat import InternVisionModel, InternVLChatModel
from transformers import AutoTokenizer

argparse = argparse.ArgumentParser()
argparse.add_argument('model_path', type=str, default='')
argparse.add_argument('vit_path', type=str, default='')

args = argparse.parse_args()

if args.model_path[-1] == '/':
    args.model_path = args.model_path[:-1]

model = InternVLChatModel.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path, trust_remote_code=True)

vit = InternVisionModel.from_pretrained(args.vit_path)
model.vision_model = vit
model.config.vision_config = vit.config
model.to(torch.bfloat16)

output_path = args.model_path + '_replace_vit'
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print('finished')
