import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.agent_utils import replay_buffer

import wandb
import copy

################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)


class DoubleDQN_CCHAIN(object):
    def __init__(self, state_shape, num_actions, device,
                 gamma=0.99, learning_rate=3e-4, hard_replacement_interval=200, memory_size=1e5, batch_size=32,
                 init_epsilon=1.0, end_epsilon=0.1, epsilon_decay_steps=100000,
                 invest=False, invest_interval=1, invest_window_size=10,
                 target_rel_loss_scale=0.01, reg_his_idx=2,
                 ):
        self.device = device
        self.in_channels = state_shape[2]
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.hr_interval = hard_replacement_interval
        self.batch_size = batch_size
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_delta = (init_epsilon - end_epsilon) / epsilon_decay_steps

        self.Q_net = QNetwork(self.in_channels, num_actions).to(self.device)
        self.Q_target = QNetwork(self.in_channels, num_actions).to(self.device)
        self.Q_target.load_state_dict(self.Q_net.state_dict())
        self.Q_net_optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=self.lr)

        self.replay_buffer = replay_buffer(buffer_size=int(memory_size))
        self.update_cnt = 0

        self.target_rel_loss_scale = target_rel_loss_scale
        self.reg_coef = 50
        self.reg_his_idx = reg_his_idx

        self.invest = invest
        self.invest_window_size = invest_window_size
        self.invest_interval = invest_interval
        # NOTE - maintain historical Q networks for investigation of policy churn
        self.frozen_init_qf = copy.deepcopy(self.Q_net)
        self.his_qf_list = [self.frozen_init_qf]

        self.q_loss_his_list = []
        self.reg_loss_his_list = []

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon - self.epsilon_delta, self.end_epsilon)

    def select_action(self, state, is_greedy=False):
        if (not is_greedy) and np.random.binomial(1, self.epsilon) == 1:
            action = np.random.choice([i for i in range(self.num_actions)])
        else:
            state = (torch.tensor(state, device=self.device).permute(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                action = self.Q_net(state).argmax(1).cpu().numpy()[0]
        return action

    def store_experience(self, s, a, r, s_, done):
        self.replay_buffer.add(s, a, r, s_, done)

    def train(self, step=None):
        # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
        batch_samples = self.replay_buffer.sample(self.batch_size)

        # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
        states = torch.FloatTensor(np.stack(batch_samples.state, axis=0)).to(self.device).permute(0, 3, 1, 2).contiguous()
        actions = torch.LongTensor(np.stack(batch_samples.action, axis=0)).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.stack(batch_samples.reward, axis=0)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(batch_samples.next_state, axis=0)).to(self.device).permute(0, 3, 1, 2).contiguous()
        dones = torch.FloatTensor(np.stack(batch_samples.is_terminal, axis=0)).to(self.device).unsqueeze(1)

        # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
        # Note: policy_network output Q-values for all the actions of a state, but all we need is the A_t taken at time t
        # in state S_t.  Thus we gather along the columns and get the Q-values corresponds to S_t, A_t.
        # Q_s_a is of size (BATCH_SIZE, 1).
        Q_sa = self.Q_net(states).gather(1, actions)

        # Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
        # Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
        # values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
        # to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

        with torch.no_grad():
            greedy_a_ = self.Q_net(next_states).argmax(1).unsqueeze(1)
        Q_sa_ = self.Q_target(next_states).detach().gather(1, greedy_a_)

        # Compute the target
        target = rewards + self.gamma * (1 - dones) * Q_sa_

        # MSE loss
        q_loss = F.mse_loss(Q_sa, target)

        # NOTE - Compute regularization loss
        if len(self.his_qf_list) < self.reg_his_idx:
            reg_loss = 0
        else:
            reg_samples = self.replay_buffer.sample(self.batch_size)
            reg_states = torch.FloatTensor(np.stack(reg_samples.state, axis=0))\
                .to(self.device).permute(0, 3, 1, 2).contiguous()
            reg_his_qf = self.his_qf_list[-self.reg_his_idx]
            reg_q = self.Q_net(reg_states)
            with torch.no_grad():
                frozen_q = reg_his_qf(reg_states)
                frozen_greedy_a = frozen_q.argmax(1).unsqueeze(1)
            reg_loss = F.mse_loss(reg_q, frozen_q)

        total_loss = q_loss + self.reg_coef * reg_loss

        # Zero gradients, backprop, update the weights of policy_net
        self.Q_net_optimizer.zero_grad()
        total_loss.backward()
        self.Q_net_optimizer.step()

        self.update_cnt += 1

        if self.update_cnt % self.hr_interval == 0:
            # Update the frozen target models
            self.Q_target.load_state_dict(self.Q_net.state_dict())

            # remain for soft replacement
            # for param, target_param in zip(self.Q_net.parameters(), self.Q_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.invest:
            # NOTE - invest policy churn for recent updates
            if self.update_cnt % 5000 == 0:
                self.invest_policy_churn(step=step)

        # NOTE - track loss scale
        self.q_loss_his_list.append(q_loss.item())
        if len(self.his_qf_list) >= self.reg_his_idx:
            self.reg_loss_his_list.append(reg_loss.item())
        if self.update_cnt >= 10000:
            # frac = 1.0 - (self.update_cnt - 1.0) / 1e6
            # frac = max(frac, 0.1)
            running_q_loss = np.mean(np.abs(self.q_loss_his_list[-50000:]))
            running_reg_loss = np.mean(self.reg_loss_his_list[-50000:])
            self.reg_coef = self.target_rel_loss_scale * running_q_loss / (running_reg_loss + 1e-8)

        # NOTE - maintain historical Q networks for investigation of policy churn
        self.his_qf_list.append(copy.deepcopy(self.Q_net))
        self.his_qf_list = self.his_qf_list[-self.invest_window_size * self.invest_interval:]

        if self.update_cnt % 5000 == 0:
            if reg_loss != 0:
                wandb.log({"losses/reg_loss": reg_loss.item()}, step=step)
            wandb.log({"losses/q_loss": q_loss.item()}, step=step)
            wandb.log({"losses/total_loss": total_loss.item()}, step=step)
            wandb.log({"auto_reg_coef": self.reg_coef}, step=step)

        return q_loss.detach().cpu().numpy()

    def invest_policy_churn(self, step):
        ref_batch_samples = self.replay_buffer.sample(self.batch_size)
        ref_states = torch.FloatTensor(np.stack(ref_batch_samples.state, axis=0))\
                     .to(self.device).permute(0, 3, 1, 2).contiguous()

        frozen_qf = self.his_qf_list[-1]
        ref_qf_list = []
        for i in range(self.invest_window_size):
            if (i + 1) * self.invest_interval > len(self.his_qf_list):
                break
            ref_qf_list.append(self.his_qf_list[-(i + 1) * self.invest_interval])
        prior_ref_q_list = []

        with torch.no_grad():
            # Calculate current greedy actions/qs
            cur_ref_q = self.Q_net(ref_states)
            cur_ref_maxq, cur_ref_greedy_a = cur_ref_q.max(1)

            # Calculate prior update greedy actions/qs for reference states
            prior_update_ref_q = frozen_qf(ref_states)
            prior_ref_q_list.append(prior_update_ref_q)

            # Calculate historical ref greedy actions/qs
            for idx, p in enumerate(ref_qf_list):
                his_ref_q = p(ref_states)
                prior_ref_q_list.append(his_ref_q)

            # Calculate outputs of frozen init policy
            init_ref_q = self.frozen_init_qf(ref_states)
            prior_ref_q_list.append(init_ref_q)

        # Calculate prior-post update diff
        cnt = 0
        # for ref_ga, ref_q in zip(prior_ref_greedy_a_list, prior_ref_q_list):
        for ref_q in prior_ref_q_list:
            # NOTE - to handle the corner case
            if cnt == 0 and self.invest_interval == 1:
                cnt += 1
                continue

            ref_maxq, ref_greedy_a = ref_q.max(1)
            invest_a_diff = ((cur_ref_greedy_a - ref_greedy_a) != 0).sum() / self.batch_size
            invest_maxq_diff = (cur_ref_maxq - ref_maxq).mean()
            invest_q_diff = (cur_ref_q - ref_q).mean()
            invest_q_diff_abs = (cur_ref_q - ref_q).abs().mean()

            if cnt == 0:
                write_suffix = ''
            elif cnt == len(prior_ref_q_list) - 1:
                write_suffix = '_init'
            else:
                write_suffix = '_' + str(cnt * self.invest_interval)
            wandb.log({"invest/greedy_a_diff" + write_suffix: invest_a_diff.item()}, step=step)
            wandb.log({"invest/maxq_diff" + write_suffix: invest_maxq_diff.item()}, step=step)
            wandb.log({"invest/q_diff_sig" + write_suffix: invest_q_diff.item()}, step=step)
            wandb.log({"invest/q_diff_abs" + write_suffix: invest_q_diff_abs.item()}, step=step)

            cnt += 1

    def save(self, filename, directory):
        torch.save(self.Q_net.state_dict(), '%s/%s_q_net.pth' % (directory, filename))
        torch.save(self.Q_target.state_dict(), '%s/%s_q_target.pth' % (directory, filename))

    def load(self, filename, directory):
        self.Q_net.load_state_dict(torch.load('%s/%s_q_net.pth' % (directory, filename)))
        self.Q_target.load_state_dict(torch.load('%s/%s_q_target.pth' % (directory, filename)))
