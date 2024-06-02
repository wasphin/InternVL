# InternVL-Chat Pre-training Evaluation

The code is modified from [OpenFlamingo Evaluation](https://github.com/mlfoundations/open_flamingo/tree/main/open_flamingo/eval)

## TODO

- [x] Initialize and develop blindly (vqa and caption)
- [x] Debug caption
- [x] Debug vqa
- [x] Align 0-shot acc
- [x] support for both multi-rounds and single-round conversation
- [x] Provide empirical results of different few-shot setting
- [ ] Update README in eval/pretrain/ (provide Usage and Options)
- [ ] \[Optional\] support classification?
- [ ] \[Optional\] support batch generate?
- [ ] Delete private sensitive information before merging

## Preparing Env

Run `pip install nltk scikit-learn inflection`

And you may need to download for nltk with:

```Python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

> Note you may need to delete spice in pycocoevalcap if is useless `$CONDA_HOME/envs/xxx/lib/pythonX.X/site-packages/pycocoevalcap/eval.py`

## Tools Usage

script `internvl_chat_dev/eval/pretrain/evaluate.py` is the python running script of evaluating pretraining.
script `internvl_chat_dev/eval/pretrain/run_eval.sh` is the shell running script of distributed evaluation for slurm user.

- `--model` : choice \["open_flamingo", "internvl_chat"\], model type
- `--result_file`: file path to save a dict of results
- `--batch_size`: we now only support batch_size=1 for internvl_chat
- `--shots`: provide multiple shots like `--shot 0 4 8 16 32`
- `--dataset`: provide multiple datasets like `--datasets flickr ok_vqa textvqa coco vqav2 vizwiz`
- `--num_trials` and `--trial_seeds` set the randomness of selecting in-context samples, the mean value will be calculated.
- `--rices` whether to use RICES for evaluation. If False, uses random demonstrations.
- `--zero-shot-add-text-shots` whether to use pure text examples when evaluating zero-shot performance, usually `--zero-shot-add-text-shots 2`
- `--chat-few-shot-style`: choice \["multi", "single"\]. whether to put all examples in the first question or conduct as a multi-rounds conversation.
- For `internvl_chat`: `--checkpoint`, `--load_in_8bit`, `--dynamic`, `--max_num` must be provided.

**Recommanded Examples**:

```
# Eval with SPOT 12 nodes
srun -p Intern5 --job-name=eval_pret --gres=gpu:8 --ntasks=12 --ntasks-per-node=1     --cpus-per-task=64  --kill-on-bad-exit=1 --quotatype=spot     bash run_eval.sh      --model internvl_chat     --checkpoint /mnt/hwfile/share_data/liqingyun/internvl_release/Mini-InternVL-Chat-2B-V1-5     --load_in_8bit False     --dynamic False     --max_num 6     --results_file "Mini-InternVL-Chat-2B-V1-5-2qa2cap-0124shots-multi_round.result"     --batch_size 1    --shots 0 1 2 4     --datasets flickr coco ok_vqa textvqa --chat-few-shot-style multi

# Debug
srun -p Intern5 --job-name=debug --gres=gpu:1 --ntasks=1 --ntasks-per-node=1     --cpus-per-task=8 --kill-on-bad-exit=1 --quotatype=spot     torchrun --standalone evaluate.py     --model internvl_chat     --checkpoint ......
```

## Coco, Flickr, OKVQA, TextVQA Results of `Mini-InternVL-Chat-2B-V1-5`

| #   |                                                  | dynamic |            shots            | okvqa | textvqa |  coco  | flickr30k |
| --- | ------------------------------------------------ | :-----: | :-------------------------: | :---: | :-----: | :----: | :-------: |
| 1   | official reported                                |    √    |              0              | 50.9  |  71.7   | 122.8  |   80.7    |
| 2   | fewshot不太可能采用 dynamic                      |         |              0              | 53.77 |  59.0   | 121.74 |   76.23   |
| 3   | 复现 #1                                          |    √    |              0              | 55.02 |  71.8   | 123.92 |   76.88   |
| 4   | #2 + two_pure_txt-shots；相对于纯0shot还是掉点的 |         | 0-two_pure_txt-multi_rounds | 50.71 |  58.27  | 119.12 |   73.58   |
| 5   | #3 + two_pure_txt-shots；相对于纯0shot还是掉点的 |    √    | 0-two_pure_txt-multi_rounds | 54.35 |  71.05  | 121.73 |   77.23   |
| 6   | #4 + single_round  prompt；掉点                  |         | 0-two_pure_txt-single_round | 50.2  |  58.26  | 112.44 |   70.08   |
| 7   | # 2 + 1-shot-multi_rounds                        |         |       1-multi_rounds        | 47.7  |  56.04  | 77.23  |   46.00   |
| 8   | # 2 + 1-shot-single_round                        |         |       1-single_round        | 47.56 |  55.79  | 66.98  |   39.85   |
| 9   | # 2 + 2-shot-multi_rounds                        |         |       2-multi_rounds        | 47.77 |  55.13  | 75.47  |   45.32   |
| 10  | # 2 + 2-shot-single_round                        |         |       2-single_round        | 47.62 |  55.28  | 64.46  |   38.49   |
| 11  | # 2 + 4-shot-multi_rounds                        |         |       4-multi_rounds        | 48.29 |  54.7   | 79.40  |   46.39   |
| 12  | # 2 + 4-shot-single_round                        |         |       4-single_round        | 46.71 |  54.45  | 61.12  |   38.94   |

______________________________________________________________________

# OpenFlamingo Evaluation Suite

This is the evaluation module of OpenFlamingo. It contains a set of utilities for evaluating multimodal models on various benchmarking datasets.

*This module is a work in progress! We will be updating this README as it develops. In the meantime, if you notice an issue, please file a Bug Report or Feature Request [here](https://github.com/mlfoundations/open_flamingo/issues/new/choose).*

## Supported datasets

| Dataset                                           | Task           | Metric         | Evaluation method |
| ------------------------------------------------- | -------------- | -------------- | ----------------- |
| [COCO](https://arxiv.org/abs/1405.0312)           | Captioning     | CIDEr          | Generation        |
| [Flickr-30K](https://aclanthology.org/Q14-1006/)  | Captioning     | CIDEr          | Generation        |
| [VQAv2](https://arxiv.org/abs/1612.00837v3)       | VQA            | VQA accuracy   | Generation        |
| [OK-VQA](https://arxiv.org/abs/1906.00067)        | VQA            | VQA accuracy   | Generation        |
| [TextVQA](https://arxiv.org/abs/1904.08920)       | VQA            | VQA accuracy   | Generation        |
| [VizWiz](https://arxiv.org/abs/1802.08218)        | VQA            | VQA accuracy   | Generation        |
| [Hateful Memes](https://arxiv.org/abs/2005.04790) | Classification | ROC AUC        | Logprobs          |
| [ImageNet](https://arxiv.org/abs/1409.0575)       | Classification | Top-1 accuracy | Logprobs          |

When evaluating a model using `num_shots` shots, we sample the exemplars from the training split. Performance is evaluated on a disjoint test split, subsampled to `--num_samples` examples (or using the full test split if `--num_samples=-1`).

## Sample scripts

Our codebase uses DistributedDataParallel to parallelize evaluation by default, so please make sure to set the `MASTER_ADDR` and `MASTER_PORT` environment variables or use `torchrun`. We provide a sample Slurm evaluation script in `open_flamingo/open_flamingo/scripts/run_eval.sh`.

We also support evaluating at a lower precision using the `--precision` flag. We find minimal difference between evaluating at full precision vs. amp_bf16.

To evaluate one of our pretrained checkpoints, we suggest first downloading a local copy of the weights, as follows:

```
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
HF_TOKEN="<your-hf-token-here>"

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
checkpoint_path= hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b",
  "checkpoint.pt",
  local_dir="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
  cache_dir="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
  local_dir_use_symlinks=False,
  token=HF_TOKEN)
print(checkpoint_path)
## openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt
```

This should place the OpenFlamingo model at the expected location in the evaluation script.

For TextVQA and VizWiz we expect annotations to be formatted differently than the original datasets. We provide the custom annotations in `open_flamingo/open_flamingo/eval/data/`. We have also uploaded all the annotation files in a [huggingface dataset](https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main) for easy access.

# Evaluating using RICES (Retrieval-based In-Context Example Selection)

We provide the option to evaluate using RICES, which is a method for selecting exemplars from the training set based on image similarity. This method was used in DeepMind's implementation for evaluating on ImageNet, but can be used for any dataset in our evaluation suite.

To use RICES, you must first create features for a benchmark's training set. We provide a script for doing so in `open_flamingo/open_flamingo/scripts/cache_rices_features.py`. This script will extract image features for a given dataset using a given CLIP model checkpoint. For example, to extract features for the COCO training set, you can run:

```bash
python cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 128 \
  --eval_coco \
  --coco_train_image_dir_path /path/to/coco/train2014 \
  --coco_val_image_dir_path /path/to/coco/val2014 \
  --coco_karpathy_json_path /path/to/coco/dataset_coco.json \
  --coco_annotations_json_path /path/to/coco/annotations/captions_train2014.json \
  --output_dir /path/to/coco/features
```

This will create a directory at `/path/to/coco/features` containing a file named `coco.pkl` with the extracted features. You can then use this directory to evaluate using RICES by passing the `--rices` flag to the evaluation script, specifying the path to the features directory using the `--cached_demonstration_features` flag, and specifying the vision encoder to use for RICES using the `--rices_vision_encoder_path` and `--rices_vision_encoder_pretrained` flags.
