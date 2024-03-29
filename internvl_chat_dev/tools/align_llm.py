
from transformers import AutoModel, AutoTokenizer

path1 = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/pretrained/internlm2-chat-20b'
path2 = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/pretrained/official_Gauss_20B_optim_V2_enhance_from_mm/50000'

output_path = '/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/pretrained/internlm2-base-20b_optim_50000'

tokenizer1 = AutoTokenizer.from_pretrained(path1, trust_remote_code=True)
tokenizer2 = AutoTokenizer.from_pretrained(path2, trust_remote_code=True)

print(tokenizer1)
print(tokenizer2)
print(f'Tokenizer2 now has {len(tokenizer2)} tokens.')

# add all special tokens from tokenizer1 to tokenizer2
added_tokens1 = [item.content for item in tokenizer1.added_tokens_decoder.values()]

# 遍历并添加到tokenizer2中
for token in added_tokens1:
    if token not in tokenizer2.all_special_tokens_extended:
        tokenizer2.add_tokens([token], special_tokens=True)
print(f'Tokenizer2 now has {len(tokenizer2)} tokens.')

# resize the tokenizer2 embeddings
model1 = AutoModel.from_pretrained(path1, trust_remote_code=True)
model2 = AutoModel.from_pretrained(path2, trust_remote_code=True)

# copy the tokenizer2 embeddings to model2
model2.resize_token_embeddings(len(tokenizer2))

# save
model2.save_pretrained(output_path)
tokenizer2.save_pretrained(output_path)

print('Done!')
