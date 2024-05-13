import torch

ckpt = torch.load('./pretrained/intern_vit_6b_448px_v1_2/mlp_projector_internlm2_7b.pth', map_location='cpu')
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

torch.save(ckpt, './pretrained/intern_vit_6b_448px_v1_2/mlp_projector_internlm2_7b.pth')
