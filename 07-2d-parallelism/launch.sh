#!/bin/bash

EXPERIMENT_NAME=tp-405b

if [ ! -f ./hosts ]; then
    echo "ERROR: ./hosts file not found. Please add this file to this current directory."
    exit 1
fi

xargs \
    -a hosts \
    -I {} \
    ssh {} \
    tmux new-session -d -s ${EXPERIMENT_NAME} -c $(pwd) \
    -e HF_HOME=/home/ubuntu/.cache/huggingface \
    -e OMP_NUM_THREADS=26 \
    $(which python) -m torch.distributed.run \
    --rdzv-id ${EXPERIMENT_NAME} \
    --rdzv-backend c10d \
    --rdzv-endpoint $(head -n 1 hosts):5001 \
    --nnodes $(grep -c '^' hosts) \
    --nproc-per-node 8 \
    --redirects 3 \
    --log-dir ./logs \
    train_llm.py \
    --experiment-name ${EXPERIMENT_NAME} \
    --dataset-name Skylion007/openwebtext \
    --model-name meta-llama/Meta-Llama-3.1-405B \
    --batch-size 1 \
    --seq-length 4096 \
    --cpu-offload on \
    --log-freq 1
