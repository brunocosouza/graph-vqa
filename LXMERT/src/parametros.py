# coding=utf-8


import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'  # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    args = {"train" : 'train', 
           "valid" : 'val',
           "test" : 'test',
           "llayers" : 9,
           "xlayers" : 5,
           "rlayers" : 5,
           "load_lxmert" : 'snap/pretrained/model',
            "load" : 'snap/output/BEST',
           "batchSize" : 32,
          "optim" : 'bert',
          "lr" : 5e-5,
          "epoch" : 200,
           "dropout" : 0.01,
          "tqdm" : True,
           "from_scratch" : False,
           "output" : 'snap/output'}
    
    args['optimizer'] =  get_optimizer(args.get("optim"))
    return args


args = parse_args()

