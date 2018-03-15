import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """ Implements the model """
    def __init__(self, config):
        super(model, self).__init__()
        self.fw_ch_rnn = nn.LSTM(
                            input_size=config.ch_em_size,
                            hidden_size=config.ch_rnn_units,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            bidirectional=False
                            )

        self.bw_ch_rnn = nn.LSTM(
                            input_size=config.ch_em_size,
                            hidden_size=config.ch_rnn_units,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            bidirectional=False
                            )

        #size of final input feature vectors. 4 is the size of Capitalization vectors.
        in_size = config.w_em_size + 2 * config.ch_rnn_units + 4
        self.w_rnn = nn.LSTM(
                            input_size=in_size,
                            hidden_size=config.w_rnn_units,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            bidirectional=True
                            )
        self.dense = nn.Linear(
                            2 * config.w_rnn_units,
                            config.w_rnn_units,
                            bias=True
                            )
        self.indp_affine = nn.Linear(
                            config.w_rnn_units,
                            config.tag_size,
                            bias=True
                            )

        self.param_init(config)
        self.embeddings()
        return
    def param_init(self, config):
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.constant(param, 0.0)
            if 'weight' in name:
                init.xavier_uniform(param)

        for name, param in self.fw_ch_rnn.named_parameters():
            if 'weight' in name:
                init.orthogonal(param)

        for name, param in self.bw_ch_rnn.named_parameters():
            if 'weight' in name:
                init.orthogonal(param)

        for name, param in self.w_rnn.named_parameters():
            if 'weight' in name:
                init.orthogonal(param)

        return
    def embeddings(self, config):
        """Add embedding layer that maps from vocabulary to vectors."""
        cap_vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        cap_lt = torch.FloatTensor(cap_vectors) #cap lookup table
        self.cap_em = nn.Embedding.from_pretrained(cap_lt, freeze=False)

        w_lt = torch.FloatTensor(config.data['word_vectors']) #word lookup table
        self.w_em = nn.Embedding.from_pretrained(w_lt, freeze=False)

        ch_lt = torch.FloatTensor(config.data['char_vectors']) #char lookup table
        self.ch_em = nn.Embedding.from_pretrained(ch_lt, freeze=False)
        return
    def map_to_variables(self, config, data_in):
        ch = torch.IntTensor(data_in['ch_in_b']) #(None, config.max_s_len, config.max_w_len)
        rev_ch = torch.IntTensor(data_in['rev_ch_in_b']) #(None, config.max_s_len, config.max_w_len)
        w = torch.IntTensor(data_in['w_in_b']) #(None, config.max_s_len)
        w_len = torch.IntTensor(data_in['w_len_b']) #(None, config.max_s_len)
        w_cap = torch.IntTensor(data_in['w_cap_b']) #(None, config.max_s_len)
        w_mask = torch.FloatTensor(data_in['w_mask_b']) #(None, config.max_s_len)
        s_len = torch.IntTensor(data_in['s_len_b']) #(None,)
        p = torch.FloatTensor(data_in['p_b']) #()
        if data_in['tag_b'] is not None:
            tag = torch.IntTensor(data_in['tag_b']) #(None, config.max_s_len)

        if torch.cuda.is_available():
            self.ch = Variable(ch.cuda())
            self.rev_ch = Variable(rev_ch.cuda())
            self.w = Variable(w.cuda())
            self.w_len = Variable(w_len.cuda())
            self.w_cap = Variable(w_cap.cuda())
            self.w_mask = Variable(w_mask.cuda())
            self.s_len = Variable(s_len.cuda())
            self.p = Variable(p.cuda())
            if data_in['tag_b'] is not None:
                self.tag = Variable(tag.cuda())
        else:
            self.ch = Variable(ch)
            self.rev_ch = Variable(rev_ch)
            self.w = Variable(w)
            self.w_len = Variable(w_len)
            self.w_cap = Variable(w_cap)
            self.w_mask = Variable(w_mask)
            self.s_len = Variable(s_len)
            self.p = Variable(p)
            if data_in['tag_b'] is not None:
                self.tag = Variable(tag)

        self.b_size = self.w.size()[0] #dynamic batch_size
        if data_in['tag_b'] is not None:
            Gpreds = torch.zeros(self.b_size * config.max_s_len, config.tag_size)
            Gpreds[torch.arange(self.b_size * config.max_s_len), self.tag.view(-1)] = 1
            Gpreds = Gpreds.view(self.b_size, config.max_s_len, config.tag_size)
            if torch.cuda.is_available():
                self.Gpreds = Variable(Gpreds.cuda())
            else:
                self.Gpreds = Variable(Gpreds)
        return
    def features(self, config):
        #bi-directional rnn to build prefix/suffix patterns.
        ch_re = self.ch_em(self.ch).view(-1, config.max_w_len, config.ch_em_size)
        fw_ch_out, _ = self.fw_ch_rnn(ch_re)

        rev_ch_re = self.ch_em(self.rev_ch).view(-1, config.max_w_len, config.ch_em_size)
        bw_ch_out, _ = self.fw_ch_rnn(rev_ch_re)

        #batch index
        b_index = torch.LongTensor(torch.arange(self.b_size)) * config.max_s_len * config.max_w_len
        b_index_re = b_index.view(self.b_size, 1)

        #sentence index
        s_index = torch.LongTensor(torch.arange(self.config.max_s_len)) * config.max_w_len
        s_index_re = s_index.view(1, config.max_s_len)

        #select last outputs for suffix/prefix patterns.
        #l_index for last character of each word
        l_index = (b_index_re + s_index_re) + (self.w_len - 1)

        #select last character's hidden state as suffix/prefix information for each word
        suffix = torch.gather(fw_ch_out.view(-1, config.ch_rnn_units), 1, l_index)
        prefix = torch.gather(bw_ch_out.view(-1, config.ch_rnn_units), 1, l_index)
        p_w_s = torch.concat((prefix, self.w_em(self.w), suffix), 2)
        p_w_s_dr = nn.Dropout(self.p)(p_w_s)

        final_features = torch.concat((p_w_s_dr, self.cap_em(self.w_cap)), 2)
        return final_features
    def encoder(self, config):
        features = self.features(config)
        h_out, _ = self.w_rnn(features)
        h_out_dr = nn.Dropout(self.p)(h_out)
        HH = h_out_dr * self.w_mask
        H = nn.Tanh(self.dense(HH))
        return H
    def indp(self, config, H):
        scores = self.indp_affine(H)
        log_probs = nn.LogSoftmax(2)(scores)
        return log_probs
    def forward(self, inputs):
        (config, data_in) = inputs
        self.map_to_variables(config, data_in)
        H = self.encoder(config)
        if config.model_type=='INDP':
            log_probs = self.indp(config, H)
        return log_probs
    def ML_loss(self, log_probs):
        objective = torch.sum(self.GPreds * log_probs, 2) * self.w_mask
        loss = -torch.mean(torch.mean(objective, 1), 2)
        return loss
