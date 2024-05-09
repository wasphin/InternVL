import torch
from internvl.model.internvl_chat import InternVisionConfig, InternVisionModel
from transformers import AutoModel

config = InternVisionConfig(
    num_channels=3,
    patch_size=14,
    image_size=448,
    qkv_bias=True,
    hidden_size=1024,
    num_attention_heads=16,
    intermediate_size=4096,
    qk_normalization=False,
    num_hidden_layers=24,
    use_flash_attn=True,
    hidden_act='gelu',
    norm_type='layer_norm',
    layer_norm_eps=1e-6,
    dropout=0.0,
    drop_path_rate=0.0,
    attention_dropout=0.0,
    initializer_range=0.02,
    initializer_factor=1.0,
)

model1 = InternVisionModel(config)
ckpt1 = model1.state_dict()
# model1.save_pretrained('./pretrained/intern_vit_300m_448px_random')
model2 = AutoModel.from_pretrained('./pretrained/clip-vit-large-patch14-336')
model2 = model2.vision_model
ckpt2 = model2.state_dict()
new_ckpt2 = {}
for k, v in ckpt2.items():
    k = k.replace('self_attn', 'attn')
    k = k.replace('layer_norm1', 'norm1')
    k = k.replace('layer_norm2', 'norm2')
    k = k.replace('out_proj', 'proj')
    if k == 'embeddings.position_embedding.weight':
        k = 'embeddings.position_embedding'
    new_ckpt2[k] = v

for i in range(24):
    q_key = f'encoder.layers.{i}.attn.q_proj.weight'
    k_key = f'encoder.layers.{i}.attn.k_proj.weight'
    v_key = f'encoder.layers.{i}.attn.v_proj.weight'
    qkv_key = f'encoder.layers.{i}.attn.qkv.weight'
    q_var = new_ckpt2[q_key]
    k_var = new_ckpt2[k_key]
    v_var = new_ckpt2[v_key]
    qkv_var = torch.cat((q_var, k_var, v_var), dim=0)
    new_ckpt2[qkv_key] = qkv_var
    del new_ckpt2[q_key]
    del new_ckpt2[k_key]
    del new_ckpt2[v_key]

    q_key = f'encoder.layers.{i}.attn.q_proj.bias'
    k_key = f'encoder.layers.{i}.attn.k_proj.bias'
    v_key = f'encoder.layers.{i}.attn.v_proj.bias'
    qkv_key = f'encoder.layers.{i}.attn.qkv.bias'
    q_var = new_ckpt2[q_key]
    k_var = new_ckpt2[k_key]
    v_var = new_ckpt2[v_key]
    qkv_var = torch.cat((q_var, k_var, v_var), dim=0)
    new_ckpt2[qkv_key] = qkv_var
    del new_ckpt2[q_key]
    del new_ckpt2[k_key]
    del new_ckpt2[v_key]

bias = ckpt1['embeddings.patch_embedding.bias']
bias = bias * 0.0
new_ckpt2['embeddings.patch_embedding.bias'] = bias
for i in range(24):
    ls1_key = f'encoder.layers.{i}.ls1'
    ls2_key = f'encoder.layers.{i}.ls2'
    new_ckpt2[ls1_key] = ckpt1[ls1_key]
    new_ckpt2[ls2_key] = ckpt1[ls2_key]

temp = new_ckpt2['embeddings.class_embedding']
temp = temp.unsqueeze(0).unsqueeze(0)
new_ckpt2['embeddings.class_embedding'] = temp

temp = new_ckpt2['embeddings.position_embedding'].unsqueeze(0)
cls = temp[:, :1, :]
pos = temp[:, 1:, :]
pos = pos.reshape(1, 24, 24, 1024)
pos = pos.permute(0, 3, 1, 2)
import torch.nn.functional as F

pos = F.interpolate(pos, size=(32, 32), mode='bicubic', align_corners=False)
pos = pos.reshape(1, 1024, 32*32).permute(0, 2, 1)
temp =  torch.cat((cls, pos), dim=1)
new_ckpt2['embeddings.position_embedding'] = temp

message = model1.load_state_dict(new_ckpt2, strict=False)
print(message)
# print("length of model1:", len(ckpt1))
# print("length of model2:", len(ckpt2))
print('difference:', set(ckpt1.keys()).symmetric_difference(set(new_ckpt2.keys())))
model1.save_pretrained('./pretrained/intern_vit_300m_448px_v1_2')
