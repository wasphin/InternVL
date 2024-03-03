set -x

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-8192}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34223

OUTPUT_DIR='work_dirs/internvl_chat_internlm2_20b_448_chinese_pretrain'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 128
# batch size per gpu: 32
# gradient accumulation steps: 2
# total batch size: 8192
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
  python -u internvl/train/internvl_chat_pretrain.py \
  --vision_path "./pretrained/intern_vit_6b_448px_v1_2" \
  --llm_path "./pretrained/internlm2-chat-20b" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/data_0218_zh_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_data_resampling False \
  --dataloader_num_workers 1 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 2e-4 \
  --weight_decay 0.05 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 384 \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
