cd eval/pretrain/

CKPT_ROOT="/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/work_dirs/interleaved/internvl_chat_v1_5_internlm2_7b_448_res_pretrain_interleaved_stage1"
RESULT_ROOT="/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/cc_results/"
CKPT_FNAMES=(
    "checkpoint-1200"
    "checkpoint-1400"
    "checkpoint-1600"
    "checkpoint-1800"
    "checkpoint-2000"
    "checkpoint-2200"
    "checkpoint-2400"
    "checkpoint-2600"
    "checkpoint-2800"
    "checkpoint-3000"
    "checkpoint-3200"
    "checkpoint-3400"
    "checkpoint-3600"
    "checkpoint-3800"
    "checkpoint-4000"
    "checkpoint-5000"
    "checkpoint-6000"
)
mkdir -p $RESULT_ROOT

for CKPT_FNAME in "${CKPT_FNAMES[@]}"; do
    set -x
    srun -p INTERN2 -w SH-IDC1-10-140-37-1 --gres=gpu:0 -N 1 \
        --cpus-per-task=2 --kill-on-bad-exit=1 --quotatype=spot \
        bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 0 1 2 4 8 --datasets coco flickr --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_01248shots-multi-rounds.result"
done
