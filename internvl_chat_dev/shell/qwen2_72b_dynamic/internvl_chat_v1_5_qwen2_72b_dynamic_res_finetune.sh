set -x

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-640}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-1280}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="/mnt/petrelfs/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/petrel-oss-python-sdk"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl_chat_v1_5/internvl_chat_v1_5_qwen2_72b_dynamic_res_finetune'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 640
# batch size per gpu: 2
# gradient accumulation steps: 1
# total batch size: 1280
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
  --model_name_or_path "./work_dirs/internvl_chat_v1_5/internvl_chat_v1_5_qwen2_72b_dynamic_res_pretrain/checkpoint-3400" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/data_yi34b_finetune_v5_33.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 6 \
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
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config_72b.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
