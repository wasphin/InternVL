import torch

ckpt = torch.load('./pretrained/InternViT-6B-448px-V1-2/mlp_projector_internlm2_7b.pth', map_location='cpu')
print(ckpt.keys())

temp = ckpt['0.weight']
temp = torch.cat((temp, temp, temp, temp), dim=0)
ckpt['0.weight'] = temp

temp = ckpt['0.bias']
temp = torch.cat((temp, temp, temp, temp), dim=0)
ckpt['0.bias'] = temp

temp = ckpt['1.weight']
temp = torch.cat((temp, temp, temp, temp), dim=1)
ckpt['1.weight'] = temp / 4.0

torch.save(ckpt, './pretrained/InternViT-6B-448px-V1-2/mlp_projector_internlm2_7b.pth')
