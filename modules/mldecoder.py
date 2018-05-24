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
    It has 2 variants for ML training:
        1-TF: Teachor Forcing
        2-SS: Scheduled Sampling
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
        """
        type = cfg.mldecoder_type
        if type=='TF':
            return self.TF_forward(H)
        elif type=='SS':
            return self.SS_forward(H)
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

        very_negative = torch.zeros(cfg.d_batch_size)
        V_Neg = Variable(very_negative.cuda()) if hasCuda else Variable(very_negative)
        V_Neg.data.fill_(-10**10)

        pads = torch.zeros(cfg.d_batch_size).long()
        Pads = Variable(pads.cuda()) if hasCuda else Variable(pads)
        Pads.data.fill_(cfg.tag_pad_id)

        lprob_candidates = torch.zeros(cfg.d_batch_size, beamsize*beamsize)
        lprob_c = Variable(lprob_candidates.cuda()) if hasCuda else Variable(lprob_candidates)

        tag_candidates = torch.zeros(cfg.d_batch_size, beamsize*beamsize).long()
        tag_c = Variable(tag_candidates.cuda()) if hasCuda else Variable(tag_candidates)

        h_candidates = torch.zeros(cfg.d_batch_size, beamsize*beamsize, cfg.dec_rnn_units)
        h_c = Variable(h_candidates.cuda()) if hasCuda else Variable(h_candidates)

        c_candidates = torch.zeros(cfg.d_batch_size, beamsize*beamsize, cfg.dec_rnn_units)
        c_c = Variable(c_candidates.cuda()) if hasCuda else Variable(c_candidates)

        beam = []
        for i in range(cfg.max_s_len):
            Hi = H[:,i,:]
            hasEnd_i = w_mask[:,i].contiguous().view(-1, 1).expand(-1, beamsize) # 1 not finished, 0 finished.
            if i==0:
                input = torch.cat((Go_symbol, Hi), dim=1)
                output, temp_c = self.dec_rnn(input, (h0, c0))
                output_H = torch.cat((output, Hi), dim=1)
                score = self.affine(output_H)
                log_prob = nn.functional.log_softmax(score, dim=1)
                log_prob.data[:, cfg.tag_pad_id] = V_Neg.data #never select pad
                kprob, kidx = torch.topk(log_prob, beamsize, dim=1, largest=True, sorted=True)

                #For the time step > 1
                h = torch.stack([output] * beamsize, dim=1)
                c = torch.stack([temp_c] * beamsize, dim=1)
                prev_tag = kidx
                prev_lprob = kprob
                beam = kidx.view(-1, beamsize, 1)


            else:
                beam_candidates = []
                prev_output = self.tag_em(prev_tag)
                for b in range(beamsize):
                    input = torch.cat((prev_output[:,b,:], Hi), dim=1)
                    output, temp_c = self.dec_rnn(input, (h[:,b,:], c[:,b,:]))
                    output_H = torch.cat((output, Hi), dim=1)
                    score = self.affine(output_H)

                    log_prob = nn.functional.log_softmax(score, dim=1)
                    log_prob.data[:, cfg.tag_pad_id] = V_Neg.data #never select pad

                    kprob, kidx = torch.topk(log_prob, beamsize, dim=1, largest=True, sorted=True)

                    for bb in range(beamsize):
                        new_lprob = prev_lprob[:,b] + hasEnd_i[:,b] * kprob[:,bb]
                        lprob_c.data[:, beamsize*b + bb] = new_lprob.data
                        tag_c.data[:, beamsize*b + bb] = (hasEnd_i[:,b].long() * kidx[:,bb] + (1.0 - hasEnd_i[:,b].long()) * Pads).data
                        h_c.data[:, beamsize*b + bb, :] = output.data
                    	c_c.data[:, beamsize*b + bb, :] = temp_c.data
                        beam_candidates.append(torch.cat((beam[:,b], tag_c[:, beamsize*b + bb].contiguous().view(-1, 1)), 1))

                    for bb in range(1, beamsize):
                        lprob_c.data[:,beamsize*b + bb] = (lprob_c[:, beamsize*b + bb] + (1.0-hasEnd_i[:,b]) * V_Neg).data

                _, maxidx = torch.topk(lprob_c, beamsize, dim=1, largest=True, sorted=True)

                beam = torch.gather(torch.stack(beam_candidates, dim=1), 1, maxidx.view(-1, beamsize, 1).expand(-1, beamsize, i+1))
                prev_tag = torch.gather(tag_c, 1, maxidx)
                prev_lprob = torch.gather(lprob_c, 1, maxidx)
                h = torch.gather(h_c, 1, maxidx.view(-1, beamsize, 1).expand(-1, beamsize, cfg.dec_rnn_units))
                c = torch.gather(c_c, 1, maxidx.view(-1, beamsize, 1).expand(-1, beamsize, cfg.dec_rnn_units))


        preds = beam
        #preds is of size (batch size, beam size, max length)
        return preds, None
