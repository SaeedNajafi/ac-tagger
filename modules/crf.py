from itertools import *
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim

hasCuda = torch.cuda.is_available()

class CRF(nn.Module):
    """
    This module uses linear-chain CRF for training, and Viterbi algorithm for decoding.
    Modified and adapted based on: https://github.com/kmkurn/pytorch-crf/blob/master/src/torchcrf/__init__.py
    """
    def __init__(self, cfg):
        super(CRF, self).__init__()

        self.cfg = cfg

        #This is a linear affine layer.
        self.affine = nn.Linear(
                            cfg.w_rnn_units,
                            cfg.tag_size,
                            bias=True
                            )

        #keeps the transition score from start symbol to the first tag.
        self.start_transitions = nn.Parameter(torch.Tensor(cfg.tag_size), requires_grad=True)

        #keeps the transition score from the last valid tag to the end symbol.
        self.end_transitions = nn.Parameter(torch.Tensor(cfg.tag_size), requires_grad=True)

        #keeps the transition score between two adjacent tags.
        self.transitions = nn.Parameter(torch.Tensor(cfg.tag_size, cfg.tag_size), requires_grad=True)

        self.param_init()
        params = ifilter(lambda p: p.requires_grad, self.parameters())
        self.opt = optim.Adam(params, lr=cfg.learning_rate)
        return

    def param_init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.constant(param, 0.0)
            if 'weight' in name:
                init.xavier_uniform(param)

        init.uniform(self.start_transitions, -0.1, 0.1)
        init.uniform(self.end_transitions, -0.1, 0.1)
        init.uniform(self.transitions, -0.1, 0.1)
        return

    def numerator_score(self, scores):
        cfg = self.cfg

        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])
        tag = Variable(cfg.B['tag'].cuda()) if hasCuda else Variable(cfg.B['tag'])

        #numerator score
        num_score = self.start_transitions[tag[:,0]]
        for i in range(cfg.max_s_len-1):
            cur_tag, next_tag = tag[:,i], tag[:,i+1]
            #Emission score for current tag
            num_score += scores[:,i,:].gather(1, cur_tag.view(-1, 1)).squeeze(1) * w_mask[:,i]
            #Transition score to next tag
            transition_score = self.transitions[cur_tag, next_tag]
            #Only add transition score if the next tag is not masked (mask == 1)
            num_score += transition_score * w_mask[:,i+1]

        s_len = Variable(cfg.B['s_len'].cuda()) if hasCuda else Variable(cfg.B['s_len'])
        last_tag_indices = s_len - 1
        last_tags = tag.gather(1, last_tag_indices.view(-1, 1)).squeeze(1)

        #End transition score
        num_score += self.end_transitions[last_tags]

        #Emission score for the last tag (time step), if mask is valid (mask == 1)
        num_score += scores[:,-1,:].gather(1, last_tags.view(-1, 1)).squeeze(1) * w_mask[:,-1,:]

        #numerator score
        return num_score

    def partition_score(self, scores):
        #http://www.cs.columbia.edu/~mcollins/fb.pdf
        cfg = self.cfg
        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])

        #Start transition score and first emission
        log_prob = self.start_transitions.view(1, -1) + scores[:,0,:]
        #Here, log_prob has size (batch_size, num_tags) where for each batch,
        #the j-th column stores the log probability that the current timestep has tag j

        for i in range(1, cfg.max_s_len):
            #Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob.unsqueeze(2)  # (batch_size, num_tags, 1)
            #Broadcast transition score over all instances in the batch
            broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            #Broadcast emission score over all possible current tags
            broadcast_emissions = emissions[:,i,:].unsqueeze(1)  # (batch_size, 1, num_tags)
            #Sum current log probability, transition, and emission scores
            score = broadcast_log_prob + broadcast_transitions + broadcast_emissions  #(batch_size, num_tags, num_tags)
            #Sum over all possible current tags, but we're in log prob space, so a sum
            #becomes a log-sum-exp
            score = self.log_sum_exp(score, 1)  # (batch_size, num_tags)
            #Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            #leave it alone
            log_prob = score * w_mask[:,i].unsqueeze(1) + log_prob * (1.0 - mask[:,i]).unsqueeze(1)

        #End transition score
        log_prob += self.end_transitions.view(1, -1)

        # Sum (log-sum-exp) over all possible tags
        return self.log_sum_exp(log_prob, 1)  # (batch_size,)

    def forward(self, H):
        #scores are emission scores for each tag at each step.
        scores = self.affine(H)
        numerator_score = self.numerator_score(scores)
        log_Z = self.partition_score(scores)
        crf_log_likelihood = numerator_score - log_Z
        return crf_log_likelihood

    def loss(self, crf_log_likelihood):
        loss = -1 * torch.sum(crf_log_likelihood, dim=0)
        return loss

    def predict(self, H):
        cfg = self.cfg

        #scores are emission scores for each tag at each step.
        scores = self.affine(H)
        s_len = Variable(cfg.B['s_len'].cuda()) if hasCuda else Variable(cfg.B['s_len'])

        emissions = scores.data
        seq_length = s_len.data

        best_tags = []
        for i in range(cfg.d_batch_size):
            emission = emissions[i]
            seq_length_i = seq_length[i]
            best_tags.append(self.viterbi_decode(emission[0:seq_length_i]))

        return best_tags

    def viterbi_decode(self, emission):
        seq_length = emission.size(0)

        #Start transition
        viterbi_score = self.start_transitions.data + emission[0]
        viterbi_path = []
        #Here, viterbi_score has shape of (num_tags,) where value at index i stores
        #the score of the best tag sequence so far that ends with tag i
        #viterbi_path saves where the best tags candidate transitioned from; this is used
        #when we trace back the best tag sequence

        #Viterbi algorithm recursive case: we compute the score of the best tag sequence
        #for every possible next tag
        for i in range(1, seq_length):
            #Broadcast viterbi score for every possible next tag
            broadcast_score = viterbi_score.view(-1, 1)
            #Broadcast emission score for every possible current tag
            broadcast_emission = emission[i].view(1, -1)
            #Compute the score matrix of shape (num_tags, num_tags) where each entry at
            #row i and column j stores the score of transitioning from tag i to tag j
            #and emitting
            score = broadcast_score + self.transitions.data + broadcast_emission
            #Find the maximum score over all possible current tag
            best_score, best_path = score.max(0)  # (num_tags,)
            #Save the score and the path
            viterbi_score = best_score
            viterbi_path.append(best_path)

        #End transition
        viterbi_score += self.end_transitions.data

        #Find the tag which maximizes the score at the last timestep; this is our best tag
        #for the last timestep
        _, best_last_tag = viterbi_score.max(0)
        best_tags = [best_last_tag[0]]

        #We trace back where the best last tag comes from, append that to our best tag
        #sequence, and trace it back again, and so on
        for path in reversed(viterbi_path):
            best_last_tag = path[best_tags[-1]]
            best_tags.append(best_last_tag)

        #Reverse the order because we start from the last timestep
        best_tags.reverse()
        return best_tags

    @staticmethod
    def log_sum_exp(tensor, dim):
        #Find the max value along `dim`
        offset, _ = tensor.max(dim)
        #Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)
        #Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
        #Add offset back
        return offset + safe_log_sum_exp
