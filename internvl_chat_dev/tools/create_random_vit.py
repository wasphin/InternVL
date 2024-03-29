from internvl.model.internvl_chat import InternVisionConfig, InternVisionModel

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
    layer_norm_eps=1e-6,
    dropout=0.0,
    drop_path_rate=0.0,
    attention_dropout=0.0,
    initializer_range=0.02,
    initializer_factor=0.1,
)

model = InternVisionModel(config)
model.save_pretrained('./pretrained/intern_vit_300m_448px_random')
