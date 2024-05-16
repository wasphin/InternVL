set -x

BATCH_SIZE=${BATCH_SIZE:-4096}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34223
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_internlm2_20b_448_dynamic_chinese_finetune_sensecore'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 128
# batch size per gpu: 4
# gradient accumulation steps: 8
# total batch size: 4096
# epoch: 1
torchrun \
  --nnodes=${WORLD_SIZE} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=8 \
  --master_port=63667 \
  python -u internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "./work_dirs/internvl_chat_internlm2_20b_448_dynamic_chinese_pretrain/checkpoint-1000_replace_llm" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/data_yi34b_finetune_v4.json" \
  --overwrite_output_dir True \
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
  --learning_rate 1e-5 \
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
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
