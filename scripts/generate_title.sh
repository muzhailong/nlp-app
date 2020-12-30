#!/bin/bash
/root/app/openmpi-3.1.2/bin/mpirun --allow-run-as-root -np 7 -H 10.18.94.4:2,10.18.94.5:2,10.18.94.6:1,10.18.94.8:2 -x LD_LIBRARY_PATH -x NCCL_IB_DISABLE=1 -x DEFAULT_NIC=enp1s0f0 -mca pml ob1 -mca btl ^openib /root/app/anaconda3/envs/python3.7.9/bin/python /home/muzhailong/nlp-app/title_generation.py \
    --train_df_path /home/muzhailong/data/weibo/train.csv \
    --base_model /home/muzhailong/nlp-app/config/title_generation_config.json \
    --checkpoint_save_path /home/muzhailong/models/title-generation-gpt2/checkpoint.pth \
    --epochs 100 \
    --batches-per-allreduce 3 \
    --batch_size 24 \
    --compression_fp16 \
    --warmup-epochs 2 \
    --save_steps 300