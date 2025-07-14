import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math


def ac_loss_clipped(new_log_probabilities, old_log_probabilities, advantages, epsilon_clip=0.2):
    probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_probabiliy_ratios = torch.clamp(
        probability_ratios, 1 - epsilon_clip, 1 + epsilon_clip
    )

    surrogate_1 = probability_ratios * advantages
    surrogate_2 = clipped_probabiliy_ratios * advantages

    return -torch.min(surrogate_1, surrogate_2)


def train_combined_networks(policy_model, value_model, combined_optimizer, data_loader, epochs=40, clip=0.2,
                            device=None):
    c1 = 0.01  # Coefficient for entropy regularization
    c2 = 0.5   # Coefficient for value loss weight

    for epoch in range(epochs):
        policy_losses = []
        value_losses = []

        for observations, actions, advantages, log_probabilities, rewards_to_go in data_loader:
            observations = observations.float().to(device)
            actions = actions.long().to(device)
            advantages = advantages.float().to(device)
            old_log_probabilities = log_probabilities.float().to(device)
            rewards_to_go = rewards_to_go.float().to(device)

            combined_optimizer.zero_grad()

            new_log_probabilities, entropy = policy_model.evaluate_actions(observations, actions)
            policy_loss = (
                ac_loss_clipped(
                    new_log_probabilities,
                    old_log_probabilities,
                    advantages,
                    epsilon_clip=clip,
                ).mean()
                - c1 * entropy.mean()
            )
            policy_losses.append(policy_loss.item())

            values = value_model(observations)
            value_loss = c2 * F.mse_loss(values, rewards_to_go)
            value_losses.append(value_loss.item())

            total_loss = policy_loss + value_loss

            total_loss.backward()
            combined_optimizer.step()


def train_combined_networks_invest(policy_model, value_model, combined_optimizer,
                                   data_loader, ref_data_loader,
                                   his_policy_list, his_value_list, his_idx=2,
                                   epochs=40, clip=0.2, device=None):
    c1 = 0.01  # Coefficient for entropy regularization
    c2 = 0.5   # Coefficient for value loss weight

    policy_losses, value_losses, reg_losses = [], [], []
    p_churn_amounts, v_churn_amounts = [], []
    train_stats_dict = {}
    for epoch in range(epochs):
        # for observations, actions, advantages, log_probabilities, rewards_to_go in data_loader:
        for train_batch, ref_batch in zip(data_loader, ref_data_loader):
            train_obs, train_a, train_adv, train_logprob, train_return = train_batch

            ref_obs, ref_a, _, _, _ = ref_batch
            ref_obs = ref_obs.float().to(device)
            ref_a = ref_a.float().to(device)

            observations = train_obs.float().to(device)
            actions = train_a.long().to(device)
            advantages = train_adv.float().to(device)
            old_log_probabilities = train_logprob.float().to(device)
            rewards_to_go = train_return.float().to(device)

            combined_optimizer.zero_grad()

            new_log_probabilities, entropy = policy_model.evaluate_actions(observations, actions)
            policy_loss = (
                ac_loss_clipped(
                    new_log_probabilities,
                    old_log_probabilities,
                    advantages,
                    epsilon_clip=clip,
                ).mean()
                - c1 * entropy.mean()
            )
            policy_losses.append(policy_loss.item())

            values = value_model(observations)
            value_loss = c2 * F.mse_loss(values, rewards_to_go)
            value_losses.append(value_loss.item())

            total_loss = policy_loss + value_loss

            # NOTE - investigate churn
            if len(his_policy_list) <= 2:
                p_churn_amounts.append(0)
                v_churn_amounts.append(0)
            else:
                ref_policy = his_policy_list[-his_idx]
                ref_vf = his_value_list[-his_idx]
                with torch.no_grad():
                    # NOTE - policy churn
                    cur_ref_action_logits, cur_ref_action_probs = policy_model.get_logits_and_probs(ref_obs)
                    ref_action_logits, ref_action_probs = ref_policy.get_logits_and_probs(ref_obs)
                    ref_p_churn = (- ref_action_probs * torch.log(cur_ref_action_probs + 1e-8)).sum(-1).mean()
                    # NOTE - value churn
                    cur_ref_value = value_model(ref_obs)
                    ref_value = ref_vf(ref_obs)
                    ref_v_churn = F.mse_loss(cur_ref_value, ref_value)
                p_churn_amounts.append(ref_p_churn.item())
                v_churn_amounts.append(ref_v_churn.item())

            total_loss.backward()
            combined_optimizer.step()

            # NOTE - add historical policies into the buffer
            his_policy_list.append(copy.deepcopy(policy_model))
            his_value_list.append(copy.deepcopy(value_model))
            # NOTE - a list with 10 policies is enough
            his_policy_list = his_policy_list[-10:]
            his_value_list = his_value_list[-10:]

    train_stats_dict['p_loss'] = np.mean(policy_losses)
    train_stats_dict['v_loss'] = np.mean(value_losses)
    train_stats_dict['reg_loss'] = np.mean(reg_losses)
    train_stats_dict['policy_churn'] = np.mean(p_churn_amounts)
    train_stats_dict['value_churn'] = np.mean(v_churn_amounts)
    return train_stats_dict, his_policy_list, his_value_list


def train_combined_networks_with_chain(policy_model, value_model, combined_optimizer,
                                       data_loader, reg_data_loader, ref_data_loader,
                                       his_policy_list, his_value_list,
                                       p_reg_coef, his_idx=2,
                                       epochs=40, clip=0.2, device=None):
    c1 = 0.01  # Coefficient for entropy regularization
    c2 = 0.5   # Coefficient for value loss weight

    policy_losses, value_losses = [], []
    p_reg_losses = []
    p_churn_amounts, v_churn_amounts = [], []
    train_stats_dict = {}
    for epoch in range(epochs):
        # for observations, actions, advantages, log_probabilities, rewards_to_go in data_loader:
        for train_batch, reg_batch, ref_batch in zip(data_loader, reg_data_loader, ref_data_loader):
            train_obs, train_a, train_adv, train_logprob, train_return = train_batch

            reg_obs, reg_a, _, _, _ = reg_batch
            reg_obs = reg_obs.float().to(device)
            reg_a = reg_a.float().to(device)
            ref_obs, ref_a, _, _, _ = ref_batch
            ref_obs = ref_obs.float().to(device)
            ref_a = ref_a.float().to(device)

            observations = train_obs.float().to(device)
            actions = train_a.long().to(device)
            advantages = train_adv.float().to(device)
            old_log_probabilities = train_logprob.float().to(device)
            rewards_to_go = train_return.float().to(device)

            combined_optimizer.zero_grad()

            new_log_probabilities, entropy = policy_model.evaluate_actions(observations, actions)
            policy_loss = (
                ac_loss_clipped(
                    new_log_probabilities,
                    old_log_probabilities,
                    advantages,
                    epsilon_clip=clip,
                ).mean()
                - c1 * entropy.mean()
            )
            policy_losses.append(policy_loss.item())

            values = value_model(observations)
            value_loss = c2 * F.mse_loss(values, rewards_to_go)
            value_losses.append(value_loss.item())

            # NOTE - compute CHAIN loss
            # NOTE - Churn reduction loss
            if len(his_policy_list) <= 2:
                p_reg_loss = 0
                p_reg_losses.append(0)
                p_churn_amounts.append(0)
                v_churn_amounts.append(0)
            else:
                reg_policy = his_policy_list[-his_idx]
                reg_vf = his_value_list[-his_idx]

                if p_reg_coef == 0:
                    p_reg_loss = 0
                    p_reg_losses.append(0)
                else:
                    cur_reg_action_logits, cur_reg_action_probs = policy_model.get_logits_and_probs(reg_obs)
                    with torch.no_grad():
                        reg_action_logits, reg_action_probs = reg_policy.get_logits_and_probs(reg_obs)
                    p_reg_loss = (- reg_action_probs * torch.log(cur_reg_action_probs + 1e-8)).sum(-1).mean()
                    p_reg_losses.append(p_reg_loss.item())

                with torch.no_grad():
                    # NOTE - policy churn
                    cur_ref_action_logits, cur_ref_action_probs = policy_model.get_logits_and_probs(ref_obs)
                    ref_action_logits, ref_action_probs = reg_policy.get_logits_and_probs(ref_obs)
                    ref_p_churn = (- ref_action_probs * torch.log(cur_ref_action_probs + 1e-8)).sum(-1).mean()
                    # NOTE - value churn
                    cur_ref_value = value_model(ref_obs)
                    ref_value = reg_vf(ref_obs)
                    ref_v_churn = F.mse_loss(cur_ref_value, ref_value)
                p_churn_amounts.append(ref_p_churn.item())
                v_churn_amounts.append(ref_v_churn.item())

            total_loss = policy_loss + value_loss
            total_loss += p_reg_coef * p_reg_loss

            total_loss.backward()
            combined_optimizer.step()

            # NOTE - add historical policies into the buffer
            his_policy_list.append(copy.deepcopy(policy_model))
            his_value_list.append(copy.deepcopy(value_model))
            # NOTE - a list with 10 policies is enough
            his_policy_list = his_policy_list[-10:]
            his_value_list = his_value_list[-10:]


    train_stats_dict['p_loss'] = np.mean(policy_losses)
    train_stats_dict['v_loss'] = np.mean(value_losses)
    train_stats_dict['p_reg_loss'] = np.mean(p_reg_losses)
    train_stats_dict['policy_churn'] = np.mean(p_churn_amounts)
    train_stats_dict['value_churn'] = np.mean(v_churn_amounts)
    return train_stats_dict, his_policy_list, his_value_list


def cumulative_sum(array, gamma=1.0):
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


class Episode:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(
        self, observation, action, reward, value, log_probability, reward_scale=20
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)
        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]


def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()


class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build_dataset(self):
        for episode in self.episodes:
            self.observations += episode.observations
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

        assert (
            len(
                {
                    len(self.observations),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probabilities),
                }
            )
            == 1
        )

        self.advantages = normalize_list(self.advantages)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )