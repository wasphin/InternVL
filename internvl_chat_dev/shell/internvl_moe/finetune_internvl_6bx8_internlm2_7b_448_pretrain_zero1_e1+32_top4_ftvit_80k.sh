set -x

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-224}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-896}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34223

export MLP_LR_SCALE=1.0
export MoE_WG_LR_SCALE=1.0
export MoE_COEFF_LR_SCALE=0.0
export VIT_NOINIT_LAYER_DECAY_RATE=1.0


export MOE_COEFF_RATIO=0.1
export JITTER_EPSILON=0.05
export COEF_LOSS_AFTER_MEAN='true'
export COEF_LINEAR_BIAS='True'

balance_loss=0.02
expert_coef_balance_coff=0.0
capacity_factor=1.2

# get the abs path of current script
SCRIPT_PATH=$(readlink -f "$0")

JOBNAME=${1:-''}

PathPrefix=shell/internvl_moe_internlm2_7b
OUTPUT_DIR=${SCRIPT_PATH//$PathPrefix/work_dirs}
OUTPUT_DIR=${OUTPUT_DIR//.sh//$JOBNAME}

# OUTPUT_DIR="work_dirs/pretrain_exp_6bx8_7b_224_balance${balance_loss}_hardcoeff0.5_lr1e-4_ep1e-4_moegate1e-4_c${capacity_factor}_dp0.0_rsamplebefore_1+32_coff.5_scale4.0_jitter0.05_coefsoftmaxwithbiasloss_newexp"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 224
# batch size per gpu: 2
# gradient accumulation steps: 2
# total batch size: 896
# epoch: 1

# export DEBUG_FLAG=True
unset PYTHONPATH
export PYTHONPATH="/mnt/petrelfs/wangweiyun/workspace_cz/InternVL/internvl_chat_dev/petrel-oss-python-sdk:${PYTHONPATH}"


srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u internvl/train/internvl_moe_chat_finetune.py \
  --model_name_or_path ./work_dirs/pretrain_internvl_6bx8_internlm2_7b_448_pretrain_zero1_e1+32_top4_ftvit_80k/checkpoint-8000 \
  --tokenizer_path /mnt/petrelfs/share_data/wangweiyun/share_ckpt/hf_home/internlm2-chat-7b \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path shell/data/internvl_sft_2m.json \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 4 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --min_dynamic_patch 1 \
  --max_dynamic_patch 6 \
  --use_thumbnail False \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_offload_config.json" \
  --report_to "tensorboard" \
  --use_moe 'deepspeedv2' \
  --moe_intermediate_size 3200 \
  --shared_expert_intermediate_size 12800 \
  --moe_shared_split_size  4 \
  --moe_output_scale 4.0 \
  --noisy_gate_policy 'RSample_before' \
  --moe_routed_expert_jitter False \
  --param_group_strategy 'v2' \
  --num_experts 32 \
  --num_routed_experts 4 \
  --num_shared_experts 1 \
  --use_weighted_residual False \
  --weighted_residual_type 'softmax' \
  --ep_size 16  \
  --use_tutel False \
  --expert_balance_coff ${balance_loss} \
  --expert_coef_balance_coff ${expert_coef_balance_coff} \
  --capacity_factor ${capacity_factor} \
  --laux_allreduce 'all_nodes' \
  --use_rts False \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

  # --resume_from_checkpoint 'work_dirs/internvl_6bx8_internlm2_7b_224_mlp_pretrain_top2gate_balance0.01_reduce_lr1e-5_moegate1e-4_c1.5_dp0.0_rsample/checkpoint-1000' \
