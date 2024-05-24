import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

path = '/mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL-Chat-V1-5-Plus'
# If your GPU has more than 40G memory, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

# calculate model size
model_size = sum(t.numel() for t in model.parameters()) / 1e6
print(f'Total model size: {model_size} MB')

vision = model.vision_model
model_size = sum(t.numel() for t in vision.parameters()) / 1e6
print(f'Vision model size: {model_size} MB')

llm = model.language_model
model_size = sum(t.numel() for t in llm.parameters()) / 1e6
print(f'LLM model size: {model_size} MB')

mlp = model.mlp1
model_size = sum(t.numel() for t in mlp.parameters()) / 1e6
print(f'MLP model size: {model_size} MB')
