from itertools import *
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

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
        self.w_rnn = nn.GRU(
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

        self.params = ifilter(lambda p: p.requires_grad, self.parameters())
        if cfg.model_type=='AC-RNN' or cfg.model_type=='BR-RNN':
            #Only for RL-training.
            self.opt = optim.SGD(self.params, lr=cfg.actor_step_size)
        else:
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

        """
        Fills the input Tensor or Variable with a (semi) orthogonal matrix,
        as described in "Exact solutions to the nonlinear dynamics of learning
        in deep linear neural networks"
        """
        #Only for RNN weights
        for name, param in self.w_rnn.named_parameters():
            if 'weight' in name:
                init.orthogonal(param)

        return

    def forward(self, F):
        cfg = self.cfg

        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])

        #Create a variable for initial hidden vector of RNNs.
        zeros = torch.zeros(2, cfg.d_batch_size, cfg.w_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        F_dr = self.drop(F)

        #Bi-directional RNN
        outputs, _ = self.w_rnn(F_dr, h0)

        outputs_dr = self.drop(outputs)

        mask = w_mask.view(cfg.d_batch_size, cfg.max_s_len, 1)
        mask_expanded = mask.expand(cfg.d_batch_size, cfg.max_s_len, 2 * cfg.w_rnn_units)
        outputs_dr_masked = outputs_dr * mask_expanded

        HH = self.dense(outputs_dr_masked)

        #tanh non-linear layer.
        #Do not change it to relu, it lowers results.
        H = nn.functional.tanh(HH)

        mask_expanded = mask.expand(cfg.d_batch_size, cfg.max_s_len, cfg.w_rnn_units)
        #H is the final matrix having final hidden vectors of steps.
        return H * mask_expanded
