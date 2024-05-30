set -x

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-4096}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl_chat_v1_5/internvl_chat_v1_5_internlm2_20b_448_res_finetune'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 128
# batch size per gpu: 16
# gradient accumulation steps: 2
# total batch size: 4096
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
  python -u internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "./work_dirs/internvl_chat_v1_5/internvl_chat_v1_5_internlm2_20b_448_res_pretrain/checkpoint-4600" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/data_0218_zh_finetune.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --unfreeze_vit_layers -2 \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 1280 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --deepspeed "zero_stage2_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
