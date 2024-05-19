set -x

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-16384}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34227
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl_chat_v1_1/internvl_chat_v1_1_husky2_13b_448_res_pretrain'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 128
# batch size per gpu: 8
# gradient accumulation steps: 16
# total batch size: 16384
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
  --model_name_or_path "./work_dirs/InternVL-Chat-ViT-6B-Husky-13B" \
  --conv_style "internvl_zh" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/data_0116_zh_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.2 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone False \
  --use_data_resampling False \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 384 \
  --do_train True \
  --grad_checkpoint False \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
