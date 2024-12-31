# This file will load a config and best model checkpoint from inference_dir and run inference, saving samples and metrics in inference_log_dir/inference_run_name
# Note: if the original training was done with ddp, this script should also use torchrun; if the original script did not use ddp, this script should not use torchrun.

torchrun --nnodes=1 \
        --nproc_per_node=8 \
        --max_restarts=0 \
        --master_port=9050 \
        --rdzv_id=100 \
        --rdzv_backend="c10d" \
        ../run.py \
                --inference_only True \
                --inference_dir "/dir_to_load" \
                --inference_log_dir "/dir_to_save" \
                --inference_run_name "inference_only"

