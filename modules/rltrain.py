import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

hasCuda = torch.cuda.is_available()

class RLTrain(nn.Module):
    """
    This module applies RL training to the decoder RNN.
    It has 3 variants:
        1- 'R': Reinforce
        2- 'BR': Reinforce with baseline
        3- 'AC': Actor-critic
    """

    def __init__(self, cfg):
        super(RLTrain, self).__init__()

        self.cfg = cfg

        #Critic:
        self.cr_size = cfg.w_rnn_units + cfg.dec_rnn_units
        self.layer1 = nn.Linear(
                            self.cr_size,
                            self.cr_size,
                            bias=True
                            )

        self.layer2 = nn.Linear(
                            self.cr_size,
                            1,
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

    #Critic approximates the state-value function of a state.
    def V(self, l, S):
        #Do not back propagate through S!
        in_S = Variable(S.data, requires_grad=False)

        #We do not apply any dropout layer as this is a regression model
        #and the optimizer will apply L2 regularization on the weights.
        H1 = nn.functional.relu(self.layer1(in_S))

        H2 =  nn.functional.sigmoid(self.layer2(H1))
        #H2 is now scaler between 0 and 1

        v = torch.div(H2, 1.0-l)
        #v is now scaler between 0 and 1.0-l which are the boundries for returns w.r.t. l and 0/1 rewards.
        return v

    #least square loss for V.
    #L2 regularization will be done by optimizer.
    def V_loss(self, Returns, prev_V):
        """
            Returns are the temporal difference or monte carlo returns calculated for
            each step. They are the target regression values for the Critic V.
            prev_V is the previous estimates of the Critic V for the returns.
            We wanna minimize the Mean Squared Error between Returns and prev_V.
        """

        #Do not back propagate through Returns!
        in_Returns = Variable(Returns.data, requires_grad=False)

        #No negative, this is MSE loss
        MSEloss = torch.mean(torch.mean(torch.pow(prev_V-in_Returns, 2.0), dim=1), dim=0)

        #mask pads
        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])

        #MSEloss will be plugged in a separate optimizer.
        return MSEloss * w_mask

    def forward(self, H, mldecoder):
        cfg = self.cfg
        dec_rnn = mldecoder.dec_rnn
        affine = mldecoder.affine
        tag_em = mldecoder.tag_em

        #Discount factor gamma
        l = cfg.gamma
        if l>=1 or l<0:
            print "INFO: 0 <= discount factor < 1 !"
            exit()

        #Actor should generate sample sequence from the model.

        #zero the pad vector
        tag_em.weight.data[cfg.tag_pad_id].fill_(0.0)

        #Create a variable for initial hidden vector of RNN.
        zeros = torch.zeros(cfg.d_batch_size, cfg.dec_rnn_units)
        h0 = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #Create a variable for the initial previous tag.
        zeros = torch.zeros(cfg.d_batch_size, cfg.tag_em_size)
        Go_symbol = Variable(zeros.cuda()) if hasCuda else Variable(zeros)

        #critic V estimates
        V_es = []

        taken_actions = []
        action_log_policies = []
        for i in range(cfg.max_s_len):
            H_i = H[:,i,:]
            if i==0:
                prev_output = Go_symbol
                h = h0

            input = torch.cat((prev_output, H_i), dim=1)

            input_dr = self.drop(input)

            output = dec_rnn(input_dr, h)

            output_dr = self.drop(output)

            output_H = torch.cat((output_dr, H_i), dim=1)

            #v = self.V(l, output_H)
            #V_es.append(v)

            score = affine(output_H)

            #For the next step
            h = output

            log_p, gen_idx = nn.functional.log_softmax(score, dim=1).max(dim=1)
            generated_prev_output = tag_em(gen_idx)
            prev_output = generated_prev_output
            taken_actions.append(gen_idx)
            action_log_policies.append(log_p)

        #V_es = torch.stack(V_es, dim=1)
        taken_actions = torch.stack(taken_actions, dim=1)
        action_log_policies = torch.stack(action_log_policies, dim=1)

        type = cfg.rltrain_type
        if type=='R':
            return self.REINFORCE(V_es, taken_actions, action_log_policies)

        elif type=='BR':
            return self.B_REINFORCE(V_es, taken_actions, action_log_policies)

        elif type=='AC':
            return self.Actor_Critic(V_es, taken_actions, action_log_policies)

        else:
            print "INFO: RLTrain type error!"
            exit()

        return None, None

    def REINFORCE(self, V_es, taken_actions, action_log_policies):
        cfg = self.cfg
        l = cfg.gamma

        tag = Variable(cfg.B['tag'].cuda()) if hasCuda else Variable(cfg.B['tag'])
        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])

        is_true_tag = torch.eq(taken_actions, tag).float()
        #0/1 reward (hamming loss) for each prediction.
        rewards = is_true_tag * w_mask

        #Monte Carlo Returns
        MC_Returns = []
        for i in range(cfg.max_s_len):
            ret = rewards[:,i].data
            for j in range(1, cfg.max_s_len - i):
                ret += (l ** j) * rewards[:,i + j].data
            MC_Returns.append(ret)

        Returns = torch.stack(MC_Returns, dim=1)
        #Do not back propagate through Returns!
        delta = Variable(Returns, requires_grad=False)
        rlloss = -torch.mean(torch.mean(action_log_policies * (delta) * w_mask, dim=1), dim=0)

        #the none is for critic loss, we do not train critic in basic REINFORCE.
        return rlloss, None

    def B_REINFORCE(self, V_es, taken_actions, action_log_policies):
        cfg = self.cfg
        l = cfg.gamma

        tag = Variable(cfg.B['tag'].cuda()) if hasCuda else Variable(cfg.B['tag'])
        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])

        is_true_tag = torch.eq(taken_actions, tag).float()
        #0/1 reward (hamming loss) for each prediction.
        rewards = is_true_tag * w_mask

        V_es = V_es * w_mask

        #Monte Carlo Returns
        MC_Returns = []
        zeros = torch.zeros(cfg.d_batch_size,)
        zeros = Variable(zeros.cuda()) if hasCuda else Variable(zeros)
        for i in range(cfg.max_s_len):
            ret = zeros
            for j in range(0, cfg.max_s_len - i):
                ret += (l ** j) * rewards[:,i + j]
            MC_Returns.append(ret)

        Returns = torch.stack(MC_Returns, dim=1)

        #Do not back propagate through Returns and V_es!
        delta = Variable(Returns.data - V_es.data, requires_grad=False)
        rlloss = -torch.mean(torch.mean(action_log_policies * (delta) * w_mask, dim=1), dim=0)

        vloss = self.V_loss(Returns, V_es)

        return rlloss, vloss

    def Actor_Critic(self, V_es, taken_actions, action_log_policies):
        cfg = self.cfg
        l = cfg.gamma
        n = cfg.n_step

        tag = Variable(cfg.B['tag'].cuda()) if hasCuda else Variable(cfg.B['tag'])
        w_mask = Variable(cfg.B['w_mask'].cuda()) if hasCuda else Variable(cfg.B['w_mask'])

        is_true_tag = torch.eq(taken_actions, tag).float()
        #0/1 reward (hamming loss) for each prediction.
        rewards = is_true_tag * w_mask

        V_es = V_es * w_mask

        #Temporal Difference Returns
        TD_Returns = []
        zeros = torch.zeros(cfg.d_batch_size,)
        zeros = Variable(zeros.cuda()) if hasCuda else Variable(zeros)
        for i in range(cfg.max_s_len):
            ret = zeros
            for j in range(n):
                if i + j < cfg.max_s_len:
                    ret += (l ** j) * rewards[:,i + j]
                    if j == n - 1:
                        if i + j + 1 < cfg.max_s_len:
                            ret += (l ** n) * V_es[:,i + n]
            TD_Returns.append(ret)

        Returns = torch.stack(TD_Returns, dim=1)

        #Do not back propagate through Returns and V_es!
        delta = Variable(Returns.data - V_es.data, requires_grad=False)
        rlloss = -torch.mean(torch.mean(action_log_policies * (delta) * w_mask, dim=1), dim=0)
        vloss = self.V_loss(Returns, V_es)

        return rlloss, vloss
