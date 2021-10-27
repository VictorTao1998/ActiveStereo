import os
import argparse
import torch
import numpy as np
import random
import os
from torch.backends import cudnn

from Sovlers import get_solver
from utils.config import cfg

def main():
    
    # Parse arguments.
    parser = argparse.ArgumentParser(description='ActiveStereoNet')

    parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                        metavar='FILE', help='Config files')
    parser.add_argument('--summary-freq', type=int, default=500, help='Frequency of saving temporary results')
    parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of saving checkpoint')
    parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
    parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')
    parser.add_argument('--warp_op', action='store_true',default=True, help='whether use warp_op function to get disparity')
    parser.add_argument('--loadmodel', type=str, help='load pretrained model')

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    args = parser.parse_args()
    
    cudnn.deterministic = False
    cudnn.benchmark = True
    
    # Create solver.
    solver = get_solver(args, cfg)
    
    # Run.
    solver.run()

if __name__ == "__main__":
    main()