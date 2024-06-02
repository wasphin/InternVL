#!/bin/bash

# proxy  # TODO Delete private sensitive information before merging
export http_proxy=http://liqingyun:yun_990608@10.1.8.50:33128/
export https_proxy=http://liqingyun:yun_990608@10.1.8.50:33128/
export HTTP_PROXY=http://liqingyun:yun_990608@10.1.8.50:33128/
export HTTPS_PROXY=http://liqingyun:yun_990608@10.1.8.50:33128/

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12322
echo $MASTER_ADDR
echo $MASTER_PORT
# SLURM_PROCID
HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo HOSTNAMES=$HOSTNAMES
H=$(hostname)
THEID=$(echo -e $HOSTNAMES | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]")
if [[ -z "${THEID// }" ]]; then
    THEID=0
fi
echo SLURM_PROCID=$THEID
NNODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo NNODES=$NNODES

set -x

torchrun --nnodes=$NNODES --nproc-per-node=8 \
    --master_port ${MASTER_PORT} --master_addr ${MASTER_ADDR} --node_rank ${THEID} \
    evaluate.py $@
