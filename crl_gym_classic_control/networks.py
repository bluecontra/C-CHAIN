import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(torch.nn.Module):
    def __init__(self, n=4, in_dim=256, activation='relu', scale_up_ratio=1, device=None):
        super(PolicyNetwork, self).__init__()

        self.device = device
        self.activation = activation
        self.scale_up_ratio = scale_up_ratio
        self.hidden_dim = 256 * scale_up_ratio

        self.fc1 = torch.nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = torch.nn.Linear(self.hidden_dim, n)

        if self.activation == 'relu':
            self.act_f = F.relu
        elif self.activation == 'l_relu':
            self.act_f = torch.nn.LeakyReLU(0.1)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))
        x = self.act_f(self.fc3(x))
        y = self.fc4(x)
        y = F.softmax(y, dim=-1)
        return y

    def sample_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(self.device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)
        dist = Categorical(y)
        action = dist.sample()
        log_probability = dist.log_prob(action)

        return action.item(), log_probability.item()

    def best_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(self.device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state).squeeze()
        action = torch.argmax(y)

        return action.item()

    def evaluate_actions(self, states, actions):
        y = self(states)
        dist = Categorical(y)
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy

    def get_logits_and_probs(self, states):
        y = self(states)
        dist = Categorical(y)
        # entropy = dist.entropy()
        probs = dist.probs

        return y, probs

    def get_logits_and_probs_detach_representation(self, states):
        # y = self(states)
        x = self.act_f(self.fc1(states))
        x = self.act_f(self.fc2(x))
        x = self.act_f(self.fc3(x))
        y = self.fc4(x.detach())
        y = F.softmax(y, dim=-1)

        dist = Categorical(y)
        # entropy = dist.entropy()
        probs = dist.probs

        return y, probs


class ValueNetwork(torch.nn.Module):
    def __init__(self, in_dim=256, activation='relu', scale_up_ratio=1, device=None):
        super(ValueNetwork, self).__init__()

        self.device = device
        self.activation = activation
        self.scale_up_ratio = scale_up_ratio
        self.hidden_dim = 256 * scale_up_ratio

        self.fc1 = torch.nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = torch.nn.Linear(self.hidden_dim, 1)
        if self.activation == 'relu':
            self.act_f = F.relu
        elif self.activation == 'l_relu':
            self.act_f = torch.nn.LeakyReLU(0.1)
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))
        x = self.act_f(self.fc3(x))
        y = self.fc4(x)

        return y.squeeze(1)

    def get_value_detach_representation(self, x):

        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))
        x = self.act_f(self.fc3(x))
        y = self.fc4(x.detach())

        return y.squeeze(1)

    def state_value(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(self.device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        return y.item()