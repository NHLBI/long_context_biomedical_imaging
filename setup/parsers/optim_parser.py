import argparse
import sys
from pathlib import Path

Setup_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Setup_DIR))

from config_utils import *

class optim_parser(object):
    """
    Parser that contains args for optimizer
    @args:
        no args
    @rets:
        no rets; self.parser contains args
    """

    def __init__(self, optim_type):
        self.parser = argparse.ArgumentParser("")
        self.parser.add_argument('--optim.lr', type=float, default=1e-4, help="Learning rate")    
        self.parser.add_argument("--optim.weight_decay", type=float, default=0.0, help='Weight decay regularization')
        if optim_type in ['adamw','adam','nadam']:
            self.add_adam_optim_args()
        if optim_type in ['lbfgs']:
            self.add_lbfgs_optim_args()

    def add_adam_optim_args(self):
        self.parser.add_argument("--optim.beta1", type=float, default=0.90, help='Beta1 for the adam optimizers')
        self.parser.add_argument("--optim.beta2", type=float, default=0.95, help='Beta2 for the adam optimizers')

    def add_lbfgs_optim_args(self):
        self.parser.add_argument("--optim.max_iter", type=int, default=100, help='maximal number of iterations per optimization step')
        self.parser.add_argument("--optim.history_size", type=int, default=100, help='update history size')
        self.parser.add_argument("--optim.line_search_fn", type=str, default=None, help='strong_wolfe or None')
