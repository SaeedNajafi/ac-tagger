from itertools import *
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init

hasCuda = torch.cuda.is_available()

class Encoder(nn.Module):
    """
    Applies a bi-direcitonl RNN
    on the input feature vectors.
    It then uses a hidden/dense layer to
    build the final hidden vectors of each step.
    """
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.cfg = cfg

        #Size of input feature vectors
        in_size = cfg.w_em_size + 2 * cfg.ch_rnn_units + cfg.cap_em_size
        self.w_rnn = nn.LSTM(
                            input_size=in_size,
                            hidden_size=cfg.w_rnn_units,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=True
                            )

        self.dense = nn.Linear(
                            2 * cfg.w_rnn_units,
                            cfg.w_rnn_units,
                            bias=True
                            )

        self.drop = nn.Dropout(cfg.dropout)

        self.param_init()
        return

    def param_init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.constant(param, 0.0)
            if 'weight' in name:
                init.xavier_uniform(param)
        return

    def forward(self, F):
        cfg = self.cfg

        #Create a variable for initial hidden vector of RNNs.
        zeros = torch.zeros(2, cfg.d_batch_size, cfg.w_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Bi-directional RNN
        outputs, _ = self.w_rnn(F, (h0, h0))

        outputs_dr = self.drop(outputs)

        HH = self.dense(outputs_dr)

        #tanh non-linear layer.
        H = nn.functional.tanh(HH)
        
        #H is the final matrix having final hidden vectors of steps.
        return H
