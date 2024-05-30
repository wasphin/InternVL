set -x

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-248}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-15872}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl_distillation_exp2'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 256
# batch size per gpu: 64
# gradient accumulation steps: 1
# total batch size: 16384
# epoch: 8
srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  --jobid 2977061 \
  ${SRUN_ARGS} \
  python -u internvl/train/internvl_distillation.py \
  --student_path "./pretrained/intern_vit_300m_448px_v1_0" \
  --teacher_path "./pretrained/InternViT-6B-448px-V1-5" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/data_0404_zh_pretrain.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --drop_path_rate 0.0 \
  --vision_select_layer -6 \
  --dataloader_num_workers 6 \
  --bf16 True \
  --num_train_epochs 8 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_steps 1000 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --do_train True \
  --grad_checkpoint False \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
