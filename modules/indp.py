from itertools import *
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim

hasCuda = torch.cuda.is_available()

class INDP(nn.Module):
    """
    This module is for independent prediciton of the tags using a softmax layer.
    """
    def __init__(self, cfg):
        super(INDP, self).__init__()

        self.cfg = cfg

        #This is a linear affine layer.
        self.affine = nn.Linear(
                            cfg.w_rnn_units,
                            cfg.tag_size,
                            bias=True
                            )
        self.param_init()

        self.params = ifilter(lambda p: p.requires_grad, self.parameters())
        self.opt = optim.Adam(self.params, lr=cfg.learning_rate)

        return


    def reset_adam(self):
        cfg = self.cfg
        self.params = ifilter(lambda p: p.requires_grad, self.parameters())
        self.opt = optim.Adam(self.params, lr=cfg.learning_rate)
        return

    def param_init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.constant(param, 0.0)
            if 'weight' in name:
                init.xavier_uniform(param)
        return

    def forward(self, H):
        scores = self.affine(H)
        log_probs = nn.functional.log_softmax(scores, dim=2)
        return log_probs

    def loss(self, log_probs):
        cfg = self.cfg
        #ML loss
        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])
        tag_o_h = Variable(cfg.B['tag_o_h'].cuda()) if hasCuda else Variable(cfg.B['tag_o_h'])

        objective = torch.sum(tag_o_h * log_probs, dim=2) * w_mask
        loss = -1 * torch.mean(torch.mean(objective, dim=1), dim=0)
        return loss

    def predict(self, H):
        log_probs = self.forward(H)
        log_p, preds = log_probs.max(dim=2)
        return preds, log_p
