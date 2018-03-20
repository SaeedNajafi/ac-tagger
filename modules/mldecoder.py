import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

hasCuda = torch.cuda.is_available()

class MLDecoder(nn.Module):
    """
    This module is for prediction of the tags using a decoder RNN.
    It has 3 variants:
        1-TF: Teachor Forcing
        2-SS: Scheduled Sampling
        3-DS: Differential Scheduled Sampling
    """

    def __init__(self, cfg):
        super(MLDecoder, self).__init__()

        self.cfg = cfg

        #Size of input feature vectors
        in_size = cfg.w_rnn_units + cfg.tag_em_size
        self.dec_rnn = nn.GRUCell(
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

        """
        Fills the input Tensor or Variable with a (semi) orthogonal matrix,
        as described in "Exact solutions to the nonlinear dynamics of learning
        in deep linear neural networks"
        """
        #Only for RNN weights
        for name, param in self.dec_rnn.named_parameters():
            if 'weight' in name:
                init.orthogonal(param)

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

            input = torch.cat((prev_output, Hi), dim=1)
            input_dr = self.drop(input)

            output = self.dec_rnn(input_dr, h)

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

        sw = Variable(switch.cuda()) if hasCuda else Variable(switch)
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

            input = torch.cat((prev_output, Hi), dim=1)
            input_dr = self.drop(input)

            output = self.dec_rnn(input_dr, h)

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
            prev_output = sw_expanded_i * generated_prev_output + (1-sw_expanded_i) * gold_prev_output

        #Return log_probs
        return nn.functional.log_softmax(torch.stack(Scores, dim=1), dim=2)

    def DS_forward(self, H):
        cfg = self.cfg

        #Sampling probability to use generated previous tag or the gold previous tag.
        sp = cfg.sampling_p

        #We feed the probability-weighted average of all tag embeddings biased strongly
        #towards the greedily generated tag.
        bias = cfg.sampling_bias

        flip_coin = torch.rand(cfg.d_batch_size, cfg.max_s_len)

        #If equal to or greater than the sampling probabiliy,
        #we will use the generated previous tag.
        switch = torch.ge(flip_coin, sp).float()

        sw = Variable(switch.cuda()) if hasCuda else Variable(switch)
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

            input = torch.cat((prev_output, Hi), dim=1)
            input_dr = self.drop(input)

            output = self.dec_rnn(input_dr, h)

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
            prev_output = sw_expanded_i * generated_prev_output + (1-sw_expanded_i) * gold_prev_output

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

            input = torch.cat((prev_output, H_i), dim=1)

            output = self.dec_rnn(input, h)

            output_H = torch.cat((output, H_i), dim=1)
            score = self.affine(output_H)
            Scores.append(score)

            #For the next step
            h = output

            _, gen_idx = nn.functional.softmax(score, dim=1).max(dim=1)
            generated_prev_output = self.tag_em(gen_idx)
            prev_output = generated_prev_output

        log_probs = nn.functional.log_softmax(torch.stack(Scores, dim=1), dim=2)
        preds = np.argmax(log_probs.cpu().data.numpy(), axis=2)
        return preds

    def beam(self, H):
        cfg = self.cfg
        beamsize = cfg.beamsize

        #zero the pad vector
        self.tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)

        #Create a variable for initial hidden vector of RNN.
        zeros = torch.zeros(cfg.d_batch_size, cfg.dec_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Create a variable for the initial previous tag.
        zeros = torch.zeros(cfg.d_batch_size, cfg.tag_em_size)
        Go_symbol = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        beam = []
        for i in range(cfg.max_s_len):
            Hi = H[:,i,:]
            if i==0:
                input = torch.cat((Go_symbol, Hi), dim=1)
                output = self.dec_rnn(input, h0)
                output_H = torch.cat((output, Hi), dim=1)
                score = self.affine(output_H)
                log_prob = nn.functional.log_softmax(score, dim=1)
                kprob, kidx = torch.topk(log_prob, beamsize, dim=1, largest=True, sorted=True)

                #For the next time step.
                h = torch.stack([output] * beamsize, dim=1)
                prev_tag = kidx
                prev_lprob = kprob

            else:
                lprob_candidates = []
                tag_candidates = []
                h_candidates = []

                prev_output = self.tag_em(prev_tag)
                for b in range(beamsize):
                    input = torch.cat((prev_output[:,b,:], Hi), dim=1)
                    output = self.dec_rnn(input, h[:,b,:])
                    output_H = torch.cat((output, Hi), dim=1)
                    score = self.affine(output_H)
                    log_prob = nn.functional.log_softmax(score, dim=1)
                    kprob, kidx = torch.topk(log_prob, beamsize, dim=1, largest=True, sorted=True)
                    h_candidates.append(output)

                    for bb in range(beamsize):
                        lprob_candidates.append(prev_lprob[:,b] + kprob[:,bb])
                        tag_candidates.append(kidx[:,bb])

                lprob_c = torch.stack(lprob_candidates, dim=1)
                prev_lprob, maxidx = torch.topk(lprob_c, beamsize, dim=1, largest=True, sorted=True)
                tag_c = torch.stack(tag_candidates, dim=1)
                new_tag = torch.gather(tag_c, 1, maxidx)
                old_tag = torch.gather(prev_tag, 1, torch.remainder(maxidx, beamsize).long())
                beam.insert(i-1, old_tag)
                beam.insert(i, new_tag)
                prev_tag = new_tag

                h_c = torch.stack(h_candidates, dim=1)
                mmaxidx = torch.remainder(maxidx, beamsize).long()
                h = torch.index_select(h_c.view(-1,cfg.dec_rnn_units), 1, mmaxidx.view(-1,)).view(-1,beamsize,cfg.dec_rnn_units)

        bm = torch.stack(beam, dim=2)
        preds = bm[:,0,:].cpu().data.numpy()
        return preds
