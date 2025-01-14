"""
Support functions for arg parsing to config
"""

import argparse
import os
import yaml

class Nestedspace(argparse.Namespace):
    """
    Define a nested namespace for dealing with nested args.
    from https://stackoverflow.com/questions/18668227/argparse-subcommands-with-nested-namespaces
    """
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

def nestedspace_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Nestedspace:
  """
  Construct a nestedspace.
  adapted from https://matthewpburruss.com/post/yaml/
  """
  return Nestedspace(**loader.construct_mapping(node))

def get_nestedspace_loader():
  """
  Add nestedspace constructors to PyYAML loader.
  adapted from https://matthewpburruss.com/post/yaml/
  """
  loader = yaml.SafeLoader
  loader.add_constructor("tag:yaml.org,2002:python/object:config_utils.Nestedspace", nestedspace_constructor)
  return loader

def config_to_yaml(config, save_path, save_name=None):
    """Save nestedspace config to yaml file."""
    if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        yaml_file_name = os.path.join(save_path,'config.yaml')
    else:
        yaml_file_name = os.path.join(save_path,f"{save_name}.yaml")
    print(f"--> save config as yaml at {yaml_file_name}")
    with open(yaml_file_name, "w", encoding = "utf-8") as yaml_file:
        dump = yaml.dump(config, default_flow_style = False, allow_unicode = True, encoding = None)
        yaml_file.write(dump)
        
    return yaml_file_name

def yaml_to_config(yaml_path, new_log_dir=None, new_run_name=None):
    """Load yaml into nestedspace config"""
    with open(yaml_path, "r") as yaml_file:
        saved_config = yaml.load(yaml_file, Loader=get_nestedspace_loader())
        if new_log_dir is not None: saved_config.log_dir = new_log_dir # Modify the log path to a new directory so we don't overwrite anything
        if new_run_name is not None: saved_config.run_name = new_run_name # Modify the log path to a new directory so we don't overwrite anything
    return saved_config

def check_for_unknown_args(config,args_to_check):
    """Check if user input any unknown args after completing all parsing"""
    unknown_args = []
    for arg in args_to_check:
        if arg not in config: 
            if '.' in arg:
                config_check = config
                for arg_element in arg.split('.'):
                    if arg_element in config_check:
                        config_check = getattr(config_check, arg_element)
                    else:
                        unknown_args += [arg]
                        break
            else: 
                unknown_args += [arg]
    return unknown_args
    
def none_or_str(value):
    """Convert arg from a string to None"""
    if value in ['None', 'none', None]:
        return None
    return value

def str_to_bool(value):
    """Convert arg from a string to bool"""
    if str(value) in ['1','True','true','T','t']: return True
    else: return False

def check_args(config):
    """Check arguments user input"""
    config.run_name = config.run_name.replace(' ','_')
    if os.path.exists(os.path.join(config.log_dir,config.run_name)) and not config.override:
        raise RuntimeError(f"User specified a log_dir ({config.log_dir}) and run_name ({config.run_name}) that already exist; either specify a different run name or override current results with --override")
    if not os.path.exists(config.data_dir):
        raise RuntimeError(f"User specified a data_dir ({config.data_dir}) that does not exist")
    if config.split_csv_path is not None:
        if not os.path.exists(config.split_csv_path):
            raise RuntimeError(f"User specified a split_csv_path ({config.split_csv_path}) that does not exist")
    if config.model_load_path is not None:
        if not os.path.exists(config.model_load_path):
            raise RuntimeError(f"User specified a model_load_path ({config.model_load_path}) that does not exist")
    if config.yaml_load_path is not None:
        if not os.path.exists(config.yaml_load_path):
            raise RuntimeError(f"User specified a yaml_load_path ({config.yaml_load_path}) that does not exist")
    if config.height <= 0:
        raise RuntimeError("Height should be greater than or equal to 1")
    if config.width <= 0:
        raise RuntimeError("Width should be greater than or equal to 1")
    if config.time <= 0:
        raise RuntimeError("Time should be greater than or equal to 1")
    if config.no_in_channel <= 0:
        raise RuntimeError("no_in_channel should be greater than or equal to 1")
    if config.no_out_channel <= 0:
        raise RuntimeError("no_out_channel should be greater than or equal to 1")
    if config.num_workers<=0: 
        config.num_workers = os.cpu_count()
    if config.prefetch_factor <= 0:
       config.prefetch_factor = 2
    if "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_WORLD_SIZE" in os.environ:
        config.ddp = True
    if config.ddp and not ("LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ or "LOCAL_WORLD_SIZE" in os.environ):
        raise RuntimeError("--ddp specified but ddp environmental variables not available; remember to run with torchrun if using ddp.")
    if not os.path.exists(config.wandb_dir): 
        os.makedirs(config.wandb_dir, exist_ok=True)
    if 'ViT' in config.encoder_name:
        assert len(config.ViT.patch_size) in [1, 3], "ViT patch size must have 1 or 3 elements"
        if len(config.ViT.patch_size)==1:
            config.ViT.patch_size = [config.ViT.patch_size[0]]*3
        assert not (config.ViT.use_hyena and config.ViT.use_mamba), "Both ViT use_hyena and use_mamba cannot be True"
    if 'Swin' in config.encoder_name:
        assert len(config.Swin.patch_size) in [1, 3], "Swin patch size must have 1 or 3 elements"
        if len(config.Swin.patch_size)==1:
            config.Swin.patch_size = [config.Swin.patch_size[0]]*3
        assert len(config.Swin.window_size) in [1, 3], "Swin window size must have 1 or 3 elements"
        if len(config.Swin.window_size)==1:
            config.Swin.window_size = [config.Swin.window_size[0]]*3
        assert not (config.Swin.use_hyena and config.Swin.use_mamba), "Both Swin use_hyena and use_mamba cannot be True"
    if config.inference_only:
        assert config.inference_dir not in [None, "None", "none"], "If inference_only is specified, user must provide an inference_dir"
        assert os.path.exists(config.inference_dir), f"inference_dir ({config.inference_dir}) does not exist"
    assert config.percent_data > 0 and config.percent_data <= 1, "percent_data must be between 0 and 1"



