set -x

PARTITION=${PARTITION:-"VC2"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/vicuna/internvl_chat_vit_6b_vicuna7b_336_pretrain'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 32
# gradient accumulation steps: 1
# total batch size: 256
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
  --vision_path "./pretrained/InternViT-6B-224px" \
  --llm_path "./pretrained/vicuna-7b-v1.5" \
  --conv_style "vicuna_v1.1" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/data_llava_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 336 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 1.0 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -4 \
  --use_data_resampling False \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 5 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 768 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size False \
  --use_thumbnail False \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
