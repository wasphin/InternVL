cd eval/pretrain/

CKPT_ROOT="/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/work_dirs/interleaved/internvl_chat_v1_5_internlm2_7b_448_res_pretrain_interleaved_stage2"
RESULT_ROOT="/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat_dev/cc_results/"
CKPT_FNAMES=(
    "checkpoint-1200"
    "checkpoint-1400"
    "checkpoint-1600"
    "checkpoint-1800"
)
mkdir -p $RESULT_ROOT

for CKPT_FNAME in "${CKPT_FNAMES[@]}"; do
    set -x
    srun -p VC2 --gres=gpu:8 -N 1 \
        --cpus-per-task=2 --kill-on-bad-exit=1 --quotatype=auto \
        bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 0 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_0shots-multi-rounds.result" &
    srun -p VC2 --gres=gpu:8 -N 1 \
        --cpus-per-task=2 --kill-on-bad-exit=1 --quotatype=auto \
        bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 1 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_1shots-multi-rounds.result" &
    srun -p VC2 --gres=gpu:8 -N 1 \
        --cpus-per-task=2 --kill-on-bad-exit=1 --quotatype=auto \
        bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 2 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_2shots-multi-rounds.result" &
    srun -p VC2 --gres=gpu:8 -N 1 \
        --cpus-per-task=2 --kill-on-bad-exit=1 --quotatype=auto \
        bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 4 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_4shots-multi-rounds.result" &
    srun -p VC2 --gres=gpu:8 -N 1 \
        --cpus-per-task=2 --kill-on-bad-exit=1 --quotatype=auto \
        bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 8 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_8shots-multi-rounds.result" &
    srun -p VC2 --gres=gpu:8 -N 1 \
        --cpus-per-task=2  --kill-on-bad-exit=1 --quotatype=auto \
        bash run_eval.sh --model internvl_chat \
        --batch_size 1 --shots 0 --datasets coco flickr ok_vqa textvqa --chat-few-shot-style multi \
        --load_in_8bit False --dynamic False --max_num 6 --zero-shot-add-text-shots 2 \
        --checkpoint $CKPT_ROOT/$CKPT_FNAME \
        --results_file "$RESULT_ROOT/results_${CKPT_FNAME}_trick0shot-multi-rounds.result" &
done
