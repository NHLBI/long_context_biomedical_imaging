import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

Project_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(Project_DIR))

from config_utils import *

class general_parser(object):
    """
    General parser that contains args used by all projects
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self):
        
        self.parser = argparse.ArgumentParser("")

        # Path args
        self.parser.add_argument("--run_name", type=str, default='project_'+str(datetime.now().strftime("%H-%M-%S-%Y%m%d")), help='Name to identify this run (in logs and wandb)')
        self.parser.add_argument("--log_dir", type=str, default=os.path.join(Project_DIR, 'logs'), help='Directory to store log files')
        self.parser.add_argument("--data_dir", type=str, default=os.path.join(Project_DIR,'data'), help='Directory where data is stored; will be passed to dataloader')
        self.parser.add_argument("--split_csv_path", type=none_or_str, default=None, help='Path to csv that specifies data splits; if not specified, data will be split into 60 percent train, 20 percent val, 20 percent test randomly (used with default dataloader only)')
        self.parser.add_argument("--model_load_path", type=none_or_str, default=None, help='Path to load full model; set to None if not loading a model')
        self.parser.add_argument("--yaml_load_path", type=none_or_str, default=None, help='Path to load yaml config from; set to None if not loading a config. Note that this config will overwrite user args.')
        self.parser.add_argument("--override", action="store_true", help="Whether to override files already saved in log_dir/run_name")
        
        # Train/eval args 
        self.parser.add_argument("--train_model", type=str_to_bool, default=True, help="Whether to run training; if False, only eval will run")
        self.parser.add_argument("--continued_training", type=str_to_bool, default=False, help="Whether to continue training; if True, will load the optimizer and scheduler states along with the model weights; used only if load_paths are specified")
        self.parser.add_argument("--eval_train_set", type=str_to_bool, default=False, help="Whether to run inference on the train set at the end of training")
        self.parser.add_argument("--eval_val_set", type=str_to_bool, default=True, help="Whether to run inference on the val set at the end of training")
        self.parser.add_argument("--eval_test_set", type=str_to_bool, default=True, help="Whether to run inference on the test set at the end of training")
        self.parser.add_argument("--save_train_samples", type=str_to_bool, default=False, help="Whether to save output samples if running inference on the train set at the end of training")
        self.parser.add_argument("--save_val_samples", type=str_to_bool, default=False, help="Whether to save output samples if running inference on the val set at the end of training")
        self.parser.add_argument("--save_test_samples", type=str_to_bool, default=True, help="Whether to save output samples if running inference on the test set at the end of training")
        
        # Inference only args
        self.parser.add_argument("--inference_only", type=str_to_bool, default=False, help="Whether to run inference only; if True, eval will run by loading the model and config from inference_dir, then running inference and saving to inference_log_dir")
        self.parser.add_argument("--inference_dir", type=str, default=None, help="Directory to load model and config from for inference_only")
        self.parser.add_argument("--inference_log_dir", type=str, default=os.path.join(Project_DIR, 'logs'), help='Directory to store log files when inference_only is true')
        self.parser.add_argument("--inference_run_name", type=str, default='project_'+str(datetime.now().strftime("%H-%M-%S-%Y%m%d")), help='Name to identify this run (in logs and wandb) when inference_only is true')

        # Wandb args
        self.parser.add_argument("--project", type=str, default='LCImaging', help='Project name for wandb')
        self.parser.add_argument("--group", type=str, default='training', help='Group for wandb')
        self.parser.add_argument("--run_notes", type=str, default='Default project notes', help='Notes for the current run for wandb')
        self.parser.add_argument("--wandb_entity", type=str, default="MyEntity", help='Wandb entity to link with')
        self.parser.add_argument("--wandb_dir", type=str, default=os.path.join(Project_DIR, 'wandb'), help='Directory for saving wandb files')
        
        # Task args
        self.parser.add_argument('--task_type', type=str, default="class", choices=['class','seg','enhance'], help="Task type for each task")
        self.parser.add_argument("--loss_func", type=str, default='CrossEntropy', choices=['CrossEntropy','MSE','CombinationEnhance'], help='Which loss function to use')
        self.parser.add_argument("--height", type=int, default=256,  help='Height (number of rows) of input; will interpolate to this (used with default dataloader only)')
        self.parser.add_argument("--width", type=int, default=256,  help='Width (number of columns) of input; will interpolate to this (used with default dataloader only)')
        self.parser.add_argument("--time", type=int, default=1,  help='Temporal/depth dimension of input; will crop/pad to this (used with default dataloader only)')
        self.parser.add_argument("--no_in_channel", type=int,  default=1, help='Number of input channels')
        self.parser.add_argument("--no_out_channel", type=int,  default=2, help='Number of output channels or classes')
        
        # Augmentation args
        self.parser.add_argument("--affine_aug", type=str_to_bool, default=True, help="Whether to apply affine transforms (used with default dataloader only)")
        self.parser.add_argument("--brightness_aug", type=str_to_bool, default=True, help="Whether to apply brightness jitter transforms (used with default dataloader only)")
        self.parser.add_argument("--gaussian_blur_aug", type=str_to_bool, default=True, help="Whether to apply gaussian blur transforms (used with default dataloader only)")

        # Model args
        self.parser.add_argument('--encoder_name', type=str, default="ViT", choices=['Identity',
                                                                                        'ViT',
                                                                                        'Swin'], 
                                                                                        help="Which encoder to use")
        self.parser.add_argument('--decoder_name', type=str, default="ViTLinear", choices=['Identity',
                                                                                      'ViTLinear',
                                                                                      'SwinLinear',
                                                                                      'UperNet2D',
                                                                                      'UperNet3D',
                                                                                      'SwinUNETR',
                                                                                      'ViTUNETR'], help="Which decoder to use")
        
        # Optimizer args
        self.parser.add_argument("--optim_type", type=str, default="adam", choices=["adam", "adamw", "nadam", "sgd", "lbfgs"],help='Which optimizer to use')
        self.parser.add_argument("--scheduler_type", type=none_or_str, default="ReduceLROnPlateau", choices=["ReduceLROnPlateau", "StepLR", "OneCycleLR", None], help='Which LR scheduler to use')
        
        # General training args
        self.parser.add_argument("--device", type=str, default='cuda', choices=['cpu','cuda'], help='Device to train on')
        self.parser.add_argument("--debug", "-D", action="store_true", help='Option to run in debug mode')
        self.parser.add_argument("--percent_data", type=float, default=1.0, help='Percentage of data to use for training')
        self.parser.add_argument("--summary_depth", type=int, default=6, help='Depth to print the model summary through')
        self.parser.add_argument("--num_workers", type=int, default=-1, help='Number of total workers for data loading; if <=0, use os.cpu_count()')
        self.parser.add_argument("--prefetch_factor", type=int, default=8, help='Number of batches loaded in advance by each worker')
        self.parser.add_argument("--use_amp", action="store_true", help='Whether to train with mixed precision')
        self.parser.add_argument("--with_timer", action="store_true", help='Whether to train with timing')
        self.parser.add_argument("--seed", type=int, default=None, help='Seed for randomization')
        self.parser.add_argument("--eval_frequency", type=int, default=1, help="How often (in epochs) to evaluate val set")
        self.parser.add_argument("--checkpoint_frequency", type=int, default=10, help="How often (in epochs) to save the model")
        self.parser.add_argument("--exact_metrics", type=str_to_bool, default=False, help="Whether to store all validation preds and gt labels to compute exact metrics, or use approximate metrics via averaging over batch; can only use with classification tasks")
        self.parser.add_argument("--ddp", action="store_true", help='Whether training with ddp; if so, call torchrun from command line')
        
        # Training parameters
        self.parser.add_argument("--num_epochs", type=int, default=50, help='Number of epochs to train for')
        self.parser.add_argument("--batch_size", type=int, default=64, help='Size of each batch')
        self.parser.add_argument("--clip_grad_norm", type=float, default=0, help='Gradient norm clip, if <=0, no clipping')
        self.parser.add_argument("--iters_to_accumulate", type=int, default=1, help='Number of iterations to accumulate gradients; if >1, gradient accumulation')
