#!/bin/bash
/root/app/openmpi-3.1.2/bin/mpirun --allow-run-as-root -np 2 -H localhost:2 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python ../title_generation.py \
    --train_df_path /home/muzhailong/data/weibo/train.csv \
    --base_model /home/muzhailong/nlp-app/config/title_generation_config.json \
    --model_save_path /home/mzl/models/title-generation-gpt2/model.pth \
    --optim_save_path /home/mzl/models/title-generation-gpt2/optim.pth \
    --epochs 100 \
    --batches-per-allreduce 3 \
    --batch_size 16 \
    --compression_fp16 \
    --warmup-epochs 2