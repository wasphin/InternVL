set -x

PARTITION=INTERN2

GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE="reserved"
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229

# number of gpus: 128
# batch size per gpu: 1
# gradient accumulation steps: 4
# total batch size: 512
# epoch: 1
srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u internvl/train/internvl_chat_with_qllama_train.py \
  --model_name_or_path "./work_dirs/1004_internvl_chat_with_qllama_vicuna13b_336_stage1" \
  --internvl_path "./data/llm/internvl_14b_224px" \
  --llm_path "./data/llm/vicuna-13b-v1.5" \
  --conv_style "vicuna_v1.1" \
  --output_dir "./work_dirs/1004_internvl_chat_with_qllama_vicuna13b_336_stage2_try1" \
  --meta_path "./shell/data/data_1214_w_pure_text.json" \
  --overwrite_output_dir True \
  --force_image_size 336 \
  --pad2square False \
  --freeze_llm True \
  --freeze_backbone True \
  --freeze_qllama True \
  --freeze_mlp True \
  --use_llm_lora 128 \
  --use_data_resampling True \
  --max_conv_num 10 \
  --with_pure_text_data True \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 400 \
  --save_total_limit 5 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --max_question_length 80 \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed "zero_stage2_config.json" \
  --report_to "tensorboard"
