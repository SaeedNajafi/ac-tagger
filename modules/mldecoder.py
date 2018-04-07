from itertools import *
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

hasCuda = torch.cuda.is_available()

class MLDecoder(nn.Module):
    """
    This module is for prediction of the tags using a decoder RNN.
    It has 3 variants for ML training:
        1-TF: Teachor Forcing
        2-SS: Scheduled Sampling
        3-DS: Differential Scheduled Sampling
    """

    def __init__(self, cfg):
        super(MLDecoder, self).__init__()

        self.cfg = cfg

        #Size of input feature vectors
        in_size = cfg.w_rnn_units + cfg.tag_em_size
        self.dec_rnn = nn.LSTMCell(
                            input_size=in_size,
                            hidden_size=cfg.dec_rnn_units,
                            bias=True
                            )

        #This is a linear affine layer.
        self.affine = nn.Linear(
                            cfg.w_rnn_units + cfg.dec_rnn_units,
                            cfg.tag_size,
                            bias=True
                            )

        self.drop = nn.Dropout(cfg.dropout)

        self.param_init()
        self.embeddings()

        return

    def param_init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.constant(param, 0.0)
            if 'weight' in name:
                init.xavier_uniform(param)
        return

    def embeddings(self):
        """Add embedding layer that maps from tag ids to tag feature vectors."""
        cfg = self.cfg
        self.tag_em = nn.Embedding(cfg.tag_size, cfg.tag_em_size)
        self.tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)
        self.tag_em.weight.requires_grad = True
        return

    def forward(self, H):
        cfg = self.cfg
        """
            Type can have three values:
            'TF': teacher force
            'SS': scheduled sampling
            'DS': differential scheduled sampling
        """
        type = cfg.mldecoder_type
        if type=='TF':
            return self.TF_forward(H)
        elif type=='SS':
            return self.SS_forward(H)
        elif type=='DS':
            return self.DS_forward(H)
        else:
            print "INFO: MLDecoder Error!"
            exit()

    def TF_forward(self, H):
        cfg = self.cfg

        #zero the pad vector
        self.tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)

        tag = Variable(cfg.B['tag'].cuda()) if hasCuda else Variable(cfg.B['tag'])


        tag_ems = self.tag_em(tag)

        #Create a variable for initial hidden vector of RNN.
        zeros = torch.zeros(cfg.d_batch_size, cfg.dec_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Create a variable for the initial previous tag.
        zeros = torch.zeros(cfg.d_batch_size, cfg.tag_em_size)
        Go_symbol = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        Scores = []
        for i in range(cfg.max_s_len):
            Hi = H[:,i,:]
            if i==0:
                prev_output = Go_symbol
                h = h0
                c = h0

            input = torch.cat((prev_output, Hi), dim=1)

            output, c = self.dec_rnn(input, (h, c))

            output_dr = self.drop(output)
            output_dr_H = torch.cat((output_dr, Hi), dim=1)

            score = self.affine(output_dr_H)
            Scores.append(score)

            #For the next step
            h = output

            #Teachor Force the previous gold tag.
            prev_output = tag_ems[:,i,:]


        #Return log_probs
        return nn.functional.log_softmax(torch.stack(Scores, dim=1), dim=2)

    def SS_forward(self, H):
        cfg = self.cfg

        #Sampling probability to use generated previous tag or the gold previous tag.
        sp = cfg.sampling_p

        flip_coin = torch.rand(cfg.d_batch_size, cfg.max_s_len)

        #If equal to or greater than the sampling probabiliy,
        #we will use the generated previous tag.
        switch = torch.ge(flip_coin, sp).float()

        sw = Variable(switch.cuda(), requires_grad=False) if hasCuda else Variable(switch, requires_grad=False)
        sw_expanded = sw.view(-1, cfg.max_s_len, 1).expand(-1, cfg.max_s_len, cfg.tag_em_size)

        #zero the pad vector
        self.tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)

        tag = Variable(cfg.B['tag'].cuda()) if hasCuda else Variable(cfg.B['tag'])
        tag_ems = self.tag_em(tag)

        #Create a variable for initial hidden vector of RNN.
        zeros = torch.zeros(cfg.d_batch_size, cfg.dec_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Create a variable for the initial previous tag.
        zeros = torch.zeros(cfg.d_batch_size, cfg.tag_em_size)
        Go_symbol = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        Scores = []
        for i in range(cfg.max_s_len):
            Hi = H[:,i,:]
            if i==0:
                prev_output = Go_symbol
                h = h0
                c = h0

            input = torch.cat((prev_output, Hi), dim=1)

            output, c = self.dec_rnn(input, (h, c))

            output_dr = self.drop(output)
            output_dr_H = torch.cat((output_dr, Hi), dim=1)
            score = self.affine(output_dr_H)
            Scores.append(score)

            #For the next step
            h = output

            #Greedily generated previous tag or the gold previous one?
            gold_prev_output = tag_ems[:,i,:]
            _, gen_idx = nn.functional.softmax(score, dim=1).max(dim=1)
            generated_prev_output = self.tag_em(gen_idx)
            sw_expanded_i = sw_expanded[:,i,:]
            prev_output = sw_expanded_i * generated_prev_output + (1.0-sw_expanded_i) * gold_prev_output

        #Return log_probs
        return nn.functional.log_softmax(torch.stack(Scores, dim=1), dim=2)

    def DS_forward(self, H):
        cfg = self.cfg

        #Sampling probability to use generated previous tag or the gold previous tag.
        sp = cfg.sampling_p

        #We feed the probability-weighted average of all tag embeddings biased strongly
        #towards the greedily generated tag.
        bias_tensor = torch.FloatTensor(1,).fill_(cfg.greedy_bias)
        bias = Variable(bias_tensor.cuda()) if hasCuda else Variable(bias_tensor)

        flip_coin = torch.rand(cfg.d_batch_size, cfg.max_s_len)

        #If equal to or greater than the sampling probabiliy,
        #we will use the generated previous tag.
        switch = torch.ge(flip_coin, sp).float()

        sw = Variable(switch.cuda(), requires_grad=False) if hasCuda else Variable(switch, requires_grad=False)
        sw_expanded = sw.view(-1, cfg.max_s_len, 1).expand(-1, cfg.max_s_len, cfg.tag_em_size)

        #zero the pad vector
        self.tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)

        tag = Variable(cfg.B['tag'].cuda()) if hasCuda else Variable(cfg.B['tag'])
        tag_ems = self.tag_em(tag)

        #Create a variable for initial hidden vector of RNN.
        zeros = torch.zeros(cfg.d_batch_size, cfg.dec_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Create a variable for the initial previous tag.
        zeros = torch.zeros(cfg.d_batch_size, cfg.tag_em_size)
        Go_symbol = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        Scores = []
        for i in range(cfg.max_s_len):
            Hi = H[:,i,:]
            if i==0:
                prev_output = Go_symbol
                h = h0
                c = h0

            input = torch.cat((prev_output, Hi), dim=1)

            output, c = self.dec_rnn(input, (h, c))

            output_dr = self.drop(output)
            output_dr_H = torch.cat((output_dr, Hi), dim=1)
            score = self.affine(output_dr_H)
            Scores.append(score)

            #For the next step
            h = output

            #Greedily generated previous tag or the gold previous one?
            gold_prev_output = tag_ems[:,i,:]


            averaging_weights = nn.functional.softmax(bias * score, dim=1)
            #Weighted average of all tag embeddings biased strongly towards the greedy best tag.
            generated_prev_output = torch.mm(averaging_weights, self.tag_em.weight)
            sw_expanded_i = sw_expanded[:,i,:]
            prev_output = sw_expanded_i * generated_prev_output + (1.0-sw_expanded_i) * gold_prev_output

        #Return log_probs
        return nn.functional.log_softmax(torch.stack(Scores, dim=1), dim=2)

    def loss(self, log_probs):
        #ML loss
        cfg = self.cfg
        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])
        tag_o_h = Variable(cfg.B['tag_o_h'].cuda()) if hasCuda else Variable(cfg.B['tag_o_h'])

        objective = torch.sum(tag_o_h * log_probs, dim=2) * w_mask
        loss = -1 * torch.mean(torch.mean(objective, dim=1), dim=0)
        return loss

    def greedy(self, H):
        cfg = self.cfg

        #zero the pad vector
        self.tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)

        #Create a variable for initial hidden vector of RNN.
        zeros = torch.zeros(cfg.d_batch_size, cfg.dec_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Create a variable for the initial previous tag.
        zeros = torch.zeros(cfg.d_batch_size, cfg.tag_em_size)
        Go_symbol = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        Scores = []
        for i in range(cfg.max_s_len):
            H_i = H[:,i,:]
            if i==0:
                prev_output = Go_symbol
                h = h0
                c = h0

            input = torch.cat((prev_output, H_i), dim=1)

            output, c = self.dec_rnn(input, (h, c))

            output_H = torch.cat((output, H_i), dim=1)
            score = self.affine(output_H)
            Scores.append(score)

            #For the next step
            h = output

            _, gen_idx = nn.functional.softmax(score, dim=1).max(dim=1)
            generated_prev_output = self.tag_em(gen_idx)
            prev_output = generated_prev_output

        log_probs = nn.functional.log_softmax(torch.stack(Scores, dim=1), dim=2)
        log_p, preds = log_probs.max(dim=2)
        return preds, log_p

    def beam(self, H):
        cfg = self.cfg
        beamsize = cfg.beamsize

        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])

        #zero the pad vector
        self.tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)

        #Create a variable for initial hidden vector of RNN.
        zeros = torch.zeros(cfg.d_batch_size, cfg.dec_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)
        c0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Create a variable for the initial previous tag.
        zeros = torch.zeros(cfg.d_batch_size, cfg.tag_em_size)
        Go_symbol = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        lprob_candidates = torch.zeros(cfg.d_batch_size, beamsize*beamsize)
        lprob_c = Variable(lprob_candidates.cuda()) if hasCuda else Variable(lprob_candidates)

        tag_candidates = torch.zeros(cfg.d_batch_size, beamsize*beamsize).long()
        tag_c = Variable(tag_candidates.cuda()) if hasCuda else Variable(tag_candidates)

        h_candidates = torch.zeros(cfg.d_batch_size, beamsize, cfg.dec_rnn_units)
        h_c = Variable(h_candidates.cuda()) if hasCuda else Variable(h_candidates)

        c_candidates = torch.zeros(cfg.d_batch_size, beamsize, cfg.dec_rnn_units)
        c_c = Variable(c_candidates.cuda()) if hasCuda else Variable(c_candidates)

        beam = []
        for i in range(cfg.max_s_len):
            Hi = H[:,i,:]
            maski = w_mask[:,i]
            if i==0:
                input = torch.cat((Go_symbol, Hi), dim=1)
                output, cc = self.dec_rnn(input, (h0, c0))
                output_H = torch.cat((output, Hi), dim=1)
                score = self.affine(output_H)
                log_prob = nn.functional.log_softmax(score, dim=1)
                kprob, kidx = torch.topk(log_prob, beamsize, dim=1, largest=True, sorted=True)

                #For the next time step.
                h = torch.stack([output] * beamsize, dim=1)
                c = torch.stack([cc] * beamsize, dim=1)

                prev_tag = kidx
                prev_lprob = kprob

            else:
                prev_output = self.tag_em(prev_tag)
                for b in range(beamsize):
                    input = torch.cat((prev_output[:,b,:], Hi), dim=1)
                    output, cc = self.dec_rnn(input, (h[:,b,:], c[:,b,:]))
                    output_H = torch.cat((output, Hi), dim=1)
                    score = self.affine(output_H)
                    log_prob = nn.functional.log_softmax(score, dim=1)
                    kprob, kidx = torch.topk(log_prob, beamsize, dim=1, largest=True, sorted=True)
                    h_c.data[:,b,:] = output.data
                    c_c.data[:,b,:] = cc.data

                    for bb in range(beamsize):
                        lprob_c.data[:,beamsize*b + bb] = (prev_lprob[:,b].data + kprob[:,bb].data * maski.data)
                        tag_c.data[:,beamsize*b + bb] = kidx[:,bb].data

                prev_lprob, maxidx = torch.topk(lprob_c, beamsize, dim=1, largest=True, sorted=True)

                """
                    which_old_ids is a trick:
                    For example in beamsize=4, maxidx can take values 0-15.
                    0-3 means in the previous time step, the tag of beam 0 was fed.
                    4-7 means in the previous time step, the tag of beam 1 was fed.
                    8-11 means in the previous time step, the tag of beam 2 was fed.
                    12-15 means in the previous time step, the tag of beam 3 was fed.
                    So maxidx//beamsize gives the beam number of previous time step.
                    Now, in this time step, we are sure which previous tag/beam/hidden was used.
                    So it should be saved. It is a backpointer saving while going
                    forward!
                """
                which_old_ids = torch.remainder(maxidx, beamsize).long()
                new_tag = torch.gather(tag_c, 1, maxidx)
                old_tag = torch.gather(prev_tag, 1, which_old_ids)
                beam.append(old_tag)
                prev_tag = new_tag
                h = torch.gather(h_c, 1, which_old_ids.view(-1, beamsize, 1).expand(-1, beamsize, cfg.dec_rnn_units))
                c = torch.gather(c_c, 1, which_old_ids.view(-1, beamsize, 1).expand(-1, beamsize, cfg.dec_rnn_units))


        beam.append(new_tag)
        preds = torch.stack(beam, dim=2)
        #!!Returning individual log probs is not implemented.!!
        #preds is of size (batch size, beam size, max length)
        return preds, None
