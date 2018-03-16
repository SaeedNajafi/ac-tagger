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
        super(Model, self).__init__()
        #self.fw_ch_rnn = nn.LSTMCell(
        #                    input_size=config.ch_em_size,
        #                    hidden_size=config.ch_rnn_units,
        #                    bias=True
        #                    )

        #self.bw_ch_rnn = nn.LSTMCell(
        #                    input_size=config.ch_em_size,
        #                    hidden_size=config.ch_rnn_units,
        #                    bias=True
        #                    )

        #size of final input feature vectors. 16 is the size of Capitalization vectors.
        #in_size = config.w_em_size + 2 * config.ch_rnn_units + 16
        self.w_rnn = nn.LSTM(
                            #input_size=in_size,
			    input_size=config.w_em_size,
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
        self.embeddings(config)
        return
    def param_init(self, config):
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.constant(param, 0.0)
            if 'weight' in name:
                init.xavier_uniform(param)
        #for name, param in self.fw_ch_rnn.named_parameters():
        #    if 'weight' in name:
        #        init.orthogonal(param)

        #for name, param in self.bw_ch_rnn.named_parameters():
        #    if 'weight' in name:
        #        init.orthogonal(param)

        for name, param in self.w_rnn.named_parameters():
            if 'weight' in name:
                init.orthogonal(param)
        return
    def embeddings(self, config):
        """Add embedding layer that maps from vocabulary to vectors."""
        #we have 4 kinds of Capitalization patterns
        #self.cap_em = nn.Embedding(4, 16)
        #self.cap_em.weight.requires_grad = True

        w_lt = torch.FloatTensor(config.data['word_vectors']) #word lookup table
        self.w_em = nn.Embedding(config.w_size, config.w_em_size)
        self.w_em.weight.data.copy_(w_lt)
        self.w_em.weight.requires_grad = True

        #ch_lt = torch.FloatTensor(config.data['char_vectors']) #char lookup table
        #self.ch_em = nn.Embedding(config.ch_size, config.ch_em_size)
        #self.ch_em.weight.data.copy_(ch_lt)
        #self.ch_em.weight.requires_grad = True
        return
    def map_to_variables(self, config, data_in):
        ch = torch.LongTensor(data_in['ch_b']) #(None, config.max_w_len)
        w_len = torch.LongTensor(data_in['w_len_b']) #(None,)
        temp = []
        for i in range(len(data_in['ch_b'])):
            each = data_in['ch_b'][i]
            lst = list(reversed(each[0:data_in['w_len_b'][i]]))
            pad_lst = [0] * (config.max_w_len - len(lst))
            lst.extend(pad_lst)
            temp.append(lst)
        rev_ch = torch.LongTensor(temp)

        w = torch.LongTensor(data_in['w_b']) #(None, config.max_s_len)
        temp = []
        for each in data_in['w_b']:
            lst = []
            for inner_each in each:
                lst.append(data_in['w_id_index'][inner_each])
            temp.append(lst)
        w_idx = torch.LongTensor(temp)
        w_cap = torch.LongTensor(data_in['w_cap_b']) #(None, config.max_s_len)

        temp = []
        for each in data_in['s_len_b']:
            lst = [1.0] * each
            pad_lst = [0.0] * (config.max_s_len-each)
            lst.extend(pad_lst)
            temp.append(lst)
        w_mask = torch.FloatTensor(temp)
        self.p = data_in['p_b']

        if data_in['tag_b'] is not None:
            tag = torch.LongTensor(data_in['tag_b']) #(None, config.max_s_len)

        if torch.cuda.is_available():
            self.ch = Variable(ch.cuda())
            self.w_len = Variable(w_len.cuda())
            self.rev_ch = Variable(rev_ch.cuda())
            self.w = Variable(w.cuda())
            self.w_idx = Variable(w_idx.cuda())
            self.w_cap = Variable(w_cap.cuda())
            self.w_mask = Variable(w_mask.cuda())
            if data_in['tag_b'] is not None:
                self.tag = Variable(tag.cuda())
        else:
            self.ch = Variable(ch)
            self.w_len = Variable(w_len)
            self.rev_ch = Variable(rev_ch)
            self.w = Variable(w)
            self.w_idx = Variable(w_idx)
            self.w_cap = Variable(w_cap)
            self.w_mask = Variable(w_mask)
            if data_in['tag_b'] is not None:
                self.tag = Variable(tag)

        self.b_size = len(data_in['w_b']) #dynamic batch_size
        if data_in['tag_b'] is not None:
            Gpreds = np.zeros((self.b_size * config.max_s_len, config.tag_size))
            Gpreds[np.arange(self.b_size * config.max_s_len), np.reshape(data_in['tag_b'], (-1,))] = 1.0
            Gpreds = np.reshape(Gpreds, (self.b_size, config.max_s_len, config.tag_size))
            if torch.cuda.is_available():
                self.Gpreds = Variable(torch.FloatTensor(Gpreds).cuda())
            else:
                self.Gpreds = Variable(torch.FloatTensor(Gpreds))
        return
    def features(self, config):
	'''
        #bi-directional rnn to build prefix/suffix patterns.
        ch = self.ch_em(self.ch).view(config.max_w_len, -1, config.ch_em_size)
        fhx = Variable(torch.zeros(ch.size()[1], ch.size()[2]).cuda())
        fcx = Variable(torch.zeros(ch.size()[1], ch.size()[2]).cuda())
        outputs = []
        for i in range(config.max_w_len):
            fhx, fcx = self.fw_ch_rnn(ch[i], (fhx, fcx))
            outputs.append(fhx)

        fw_ch_h = torch.stack(outputs, dim=1)

        #l_index for last character of each word
        l_index = (self.w_len - 1).view(-1, 1, 1).expand(-1, 1, config.ch_rnn_units)
        suffix = torch.gather(fw_ch_h, 1, l_index).view(-1, config.ch_rnn_units)


        rev_ch = self.ch_em(self.rev_ch).view(config.max_w_len, -1, config.ch_em_size)
        bhx = Variable(torch.zeros(rev_ch.size()[1], rev_ch.size()[2]).cuda())
        bcx = Variable(torch.zeros(rev_ch.size()[1], rev_ch.size()[2]).cuda())
        outputs = []
        for i in range(config.max_w_len):
            bhx, bcx = self.bw_ch_rnn(rev_ch[i], (bhx, bcx))
            outputs.append(bhx)

        bw_ch_h = torch.stack(outputs, dim=1)
        prefix = torch.gather(bw_ch_h, 1, l_index).view(-1, config.ch_rnn_units)


        S = torch.index_select(suffix, 0, self.w_idx.view(-1,)).view(-1, config.max_s_len, config.ch_rnn_units)
        P = torch.index_select(prefix, 0, self.w_idx.view(-1,)).view(-1, config.max_s_len, config.ch_rnn_units)
        W = self.w_em(self.w)
        C = self.cap_em(self.w_cap)
        p_w_s_c = torch.cat((P, W, S, C), 2)
        final_features = nn.Dropout(self.p)(p_w_s_c)
	'''
	final_features = self.w_em(self.w)
        return final_features
    def encoder(self, config):
        features = self.features(config)
        h_out, _ = self.w_rnn(features)
        h_out_dr = nn.Dropout(self.p)(h_out)
        #mask = self.w_mask.view(-1, config.max_s_len, 1).expand(-1, config.max_s_len, 2 * config.w_rnn_units)
        #HH = h_out_dr * mask
	HH = h_out_dr
        H = nn.Tanh()(self.dense(HH))
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
        objective = torch.sum(self.Gpreds * log_probs, dim=2) * self.w_mask
        loss = -torch.mean(torch.mean(objective, dim=1), dim=0)
        return loss
