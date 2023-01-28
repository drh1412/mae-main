import os

# os.system('python submitit_finetune.py \
#     --job_dir /media/drh/DATA/calcium/mae-main/filetestfinetune \
#     --nodes 1 \
#     --batch_size 1 \
#     --model vit_large_patch16 \
#     --finetune  /media/drh/DATA/calcium/mae-main/mae-main/demo/mae_finetuned_vit_large.pth\
#     --epochs 2 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval --data_path /media/drh/DATA/calcium/mae-main/dataoftest')
os.system('python main_pretrain.py \
    --batch_size 32 \
    --model  mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 25000 \
    --warmup_epochs 0 \
    --blr 1.5e-3 --weight_decay 0.05 \
    --data_path /share/home/dongrh/mae/dataoftest')
