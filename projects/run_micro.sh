export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=1 \
        --nproc_per_node=8 \
        --max_restarts=0 \
        --master_port=9050 \
        --rdzv_id=100 \
        --rdzv_backend="c10d" \
        ../run.py \
                --run_name="microscopy_denoising" \
                --project='lc_biomedical_img' \
                --wandb_entity='my_wandb_entity' \
                --data_dir="preprocessed_data/micro" \
                --split_csv_path="csv_samplers/micro_split.csv" \
                --task_type=enhance \
                --exact_metrics=False \
                --height=1024 \
                --width=1024 \
                --time=1 \
                --no_in_channel=1 \
                --no_out_channel=1 \
                --affine_aug=True \
                --brightness_aug=True \
                --gaussian_blur_aug=False \
                --batch_size 4 \
                --num_epochs=250 \
                --train_model=True \
                --encoder_name=Swin \
                --Swin.size='tiny' \
                --Swin.patch_size 2 \
                --Swin.window_size 4 \
                --Swin.use_hyena False \
                --Swin.use_mamba True \
                --decoder_name=UperNet2D \
                --loss_func=CombinationEnhance \
                --optim_type=adam \
                --optim.lr=1e-5 \
                --optim.beta1=0.9 \
                --optim.beta2=0.99 \
                --scheduler_type=OneCycleLR \
                --device=cuda \
                --num_workers=16 \
                --seed 1 \
                --checkpoint_frequency 500 \
                --override \

