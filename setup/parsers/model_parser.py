
import argparse
import sys
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

from config_utils import *

class model_parser(object):
    """
    Parser that contains args for the model architecture
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self, model_type):
        self.parser = argparse.ArgumentParser("")

        if 'ViT' in model_type:
            self.add_ViT_args()
        
        if 'Swin' in model_type:
            self.add_SWIN_args()
        
    def add_ViT_args(self): 
        self.parser.add_argument('--ViT.size', type=str, default='small', choices=['small','base','custom'], help="Size of ViT model")
        self.parser.add_argument('--ViT.patch_size', nargs='+', type=int, default=[16,16,16], help="Size of ViT patches (number of pixels per token), ordered as T, H, W")
        self.parser.add_argument('--ViT.hidden_size', type=int, default=768, help="Size of embedding dimension")
        self.parser.add_argument('--ViT.mlp_dim', type=int, default=3072, help="Size of mlp dimension")
        self.parser.add_argument('--ViT.num_layers', type=int, default=12, help="Number of transformer blocks")
        self.parser.add_argument('--ViT.num_heads', type=int, default=12, help="Number of attention heads")
        self.parser.add_argument('--ViT.use_hyena', type=str_to_bool, default=False, help="Whether to use hyena in place of attention block")
        self.parser.add_argument('--ViT.use_mamba', type=str_to_bool, default=False, help="Whether to use mamba in place of attention block")

    def add_SWIN_args(self):  
        self.parser.add_argument('--Swin.size', type=str, default='tiny', choices=['unetr','tiny','small','base','large','custom'], help="Size of SWIN model")
        self.parser.add_argument('--Swin.patch_size', nargs='+', type=int, default=[2,2,2], help="Size of swin patches (number of pixels per token), ordered as T, H, W")
        self.parser.add_argument('--Swin.window_size', nargs='+', type=int, default=[8,8,8], help="Size of swin windows (number of tokens per attn window), ordered as T, H, W")
        self.parser.add_argument('--Swin.embed_dim', type=int, default=24, help="Size of embedding dimension")
        self.parser.add_argument('--Swin.depths', nargs='+', type=int, default=[2,2,6,2], help="Number of transformer blocks per resolution depth")
        self.parser.add_argument('--Swin.num_heads', nargs='+', type=int, default=[3,6,12,24], help="Number of attention heads per resolution depth")
        self.parser.add_argument('--Swin.use_hyena', type=str_to_bool, default=False, help="Whether to use hyena in place of attention block")
        self.parser.add_argument('--Swin.use_mamba', type=str_to_bool, default=False, help="Whether to use mamba in place of attention block")
