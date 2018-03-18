import torch
import torch.nn as nn
from torch.nn import init

class INDP(nn.Module):
    """
    This module is for independent prediciton of the tags using a softmax layer.
    """
    def __init__(self, cfg):
        super(INDP, self).__init__()

        #This is a linear affine layer.
        self.affine = nn.Linear(
                            cfg.w_rnn_units,
                            cfg.tag_size,
                            bias=True
                            )
        self.param_init()
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
