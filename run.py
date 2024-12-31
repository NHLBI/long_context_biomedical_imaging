"""
Standard run file 
"""

import os, sys, glob
from setup.setup_base import parse_config_and_setup_run
from setup.config_utils import config_to_yaml
from model.model_base import EncoderDecoderModel
from model.model_utils import save_model, load_model
from optim.optim_base import OptimManager
from trainer.trainer_base import TrainManager
from metrics.metrics_base import MetricManager
from loss.loss_base import get_loss_func 
from data.data_base import NumpyDataset

# -------------------------------------------------------------------------------------------------
def main():
    
    # -----------------------------
    # Parse input arguments to config
    config = parse_config_and_setup_run()
    
    # -----------------------------
    # Define datasets
    train_set = NumpyDataset(config=config, split='train')
    val_set = NumpyDataset(config=config, split='val')
    test_set = NumpyDataset(config=config, split='test')
    
    # -----------------------------
    # Create the model
    model = EncoderDecoderModel(config=config, 
                                encoder_name=config.encoder_name, 
                                decoder_name=config.decoder_name, 
                                input_feature_channels=config.no_in_channel, 
                                output_feature_channels=config.no_out_channel)
    
    # -----------------------------
    # Load model weights, if a load path is specified
    if config.model_load_path is not None:
        model = load_model(model, config.model_load_path, device=config.device)

    # -----------------------------
    # Create the loss function
    loss_func = get_loss_func(config.loss_func) 

    # -----------------------------
    # Create OptimManager, which defines optimizers and schedulers
    optim_manager = OptimManager(config, model, train_set)

    # -----------------------------
    # Create MetricManager, which tracks metrics and checkpoints models during training
    metric_manager = MetricManager(config)

    # -----------------------------
    # Create TrainManager, which will control model training
    train_manager = TrainManager(config=config,
                                 datasets=[train_set, val_set, test_set],
                                 model=model,
                                 loss_func=loss_func,
                                 optim_manager=optim_manager,
                                 metric_manager=metric_manager)
    
    # -----------------------------
    # Save config to yaml file
    yaml_file = config_to_yaml(config,os.path.join(config.log_dir,config.run_name))
    config.yaml_file = yaml_file 

    # -----------------------------
    # Execute training and evaluation
    train_manager.run()

    

# -------------------------------------------------------------------------------------------------
if __name__=="__main__":    
    main()



