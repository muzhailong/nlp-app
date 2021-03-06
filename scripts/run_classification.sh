/root/app/openmpi-3.1.2/bin/mpirun --allow-run-as-root -np 2 -H localhost:2 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python ../bert_classification.py \
    --train_df_path /home/muzhailong/data/tnews_public/train.csv \
    --eval_df_path /home/muzhailong/data/tnews_public/dev.csv \
    --predict_df_path /home/muzhailong/data/tnews_public/test.csv \
    --model_save_path /home/muzhailong/models/tnews/model.pth \
    --optim_save_path /home/muzhailong/models/tnews/optim.pth \
    --do_train \
    --do_eval \
    --do_predict \
    --epochs 100 \
    --batchs_per_step 3 \
    --compression_fp16 \
    --batch_size 32