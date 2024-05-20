from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './pretrained/Meta-Llama-3-8B'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

num_new_tokens = tokenizer.add_tokens([
    '<|system|>', '<|user|>', '<|assistant|>', '<|end|>', '<|pad|>'
], special_tokens=True)
tokenizer.pad_token = '<|pad|>'
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.eos_token = '<|end|>'
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
print('padding_side', tokenizer.padding_side)

print(f'num_new_tokens: {num_new_tokens}')
model.resize_token_embeddings(len(tokenizer))
output_embeddings = model.get_output_embeddings().weight.data
output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
output_embeddings[-num_new_tokens:] = output_embeddings_avg

model.config.vocab_size = len(tokenizer)

model.save_pretrained('./pretrained/Meta-Llama-3-8B-Add-Token')
print('save model success')
tokenizer.save_pretrained('./pretrained/Meta-Llama-3-8B-Add-Token')
print('save tokenizer success')
