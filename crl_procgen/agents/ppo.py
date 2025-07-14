from .base_agent import BaseAgent
from common.env.procgen_wrappers import *
import torch
import numpy as np
from procgen import ProcgenEnv
# from trac import start_trac
import copy
from collections import OrderedDict
import math

from agents.redo_utils import redo_reset


def environment_generator_procgen(env_name, total_levels, seed=0, n_envs=1, distribution_mode="hard"):
    for current_level in range(total_levels):
        current_level += seed
        env = ProcgenEnv(num_envs=n_envs, env_name=env_name, start_level=current_level, num_levels=1,
                         distribution_mode=distribution_mode)
        env = VecExtractDictObs(env, "rgb")
        env = VecNormalize(env, ob=False)
        env = TransposeFrame(env)
        env = ScaledFloatFrame(env)
        yield env


class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=1000,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=2048,
                 gamma=0.999,
                 lmbda=0.95,
                 learning_rate=0.001,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device, n_checkpoints)

        self.n_steps = n_steps
        self.n_envs = 1
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        # self.optimizer = None
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

        self.his_policy_list = [copy.deepcopy(self.policy)]

        # NOTE -used for ReDO
        self.total_epoch_count = 0

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def train(self, optimizer):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for _ in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def train_with_churn_reduction(self, optimizer, p_reg_coef, chain_batch_ratio=1, his_idx=1):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        p_reg_losses = []
        p_churn_amounts, v_churn_amounts = [], []

        self.policy.train()
        for _ in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            ref_data_generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            reg_data_generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size * chain_batch_ratio)
            for sample, ref_sample, reg_sample in zip(generator, ref_data_generator, reg_data_generator):
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                entropy_loss = dist_batch.entropy().mean()

                # NOTE - compute churn reduction loss
                if len(self.his_policy_list) <= his_idx + 1:
                    p_reg_loss = 0
                    p_reg_losses.append(0)
                    p_churn_amounts.append(0)
                    v_churn_amounts.append(0)
                else:
                    reg_policy = self.his_policy_list[-his_idx - 1]
                    ref_policy = self.his_policy_list[-his_idx - 1]

                    reg_obs_batch, _, _, _, _, _, _, _ = reg_sample
                    cur_reg_dist_batch, cur_reg_value_batch, _ = self.policy(reg_obs_batch, None, None)
                    with torch.no_grad():
                        reg_dist_batch, reg_value_batch, _ = reg_policy(reg_obs_batch, None, None)

                    if p_reg_coef == 0:
                        p_reg_loss = 0
                        p_reg_losses.append(0)
                    else:
                        p_reg_loss = (- reg_dist_batch.probs * torch.log(cur_reg_dist_batch.probs + 1e-8)).sum(-1).mean()
                        p_reg_losses.append(p_reg_loss.item())

                    with torch.no_grad():
                        ref_obs_batch, _, _, _, _, _, _, _ = ref_sample
                        cur_ref_dist_batch, cur_ref_value_batch, _ = self.policy(ref_obs_batch, None, None)
                        ref_dist_batch, ref_value_batch, _ = ref_policy(ref_obs_batch, None, None)

                        ref_p_churn = (- ref_dist_batch.probs * torch.log(cur_ref_dist_batch.probs + 1e-8)).sum(-1).mean()
                        ref_v_churn = (cur_ref_value_batch - ref_value_batch).pow(2).mean()
                    p_churn_amounts.append(ref_p_churn.item())
                    v_churn_amounts.append(ref_v_churn.item())

                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss += p_reg_coef * p_reg_loss
                loss.backward()

                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

                # NOTE - save historical policies/vfs
                # NOTE - add historical policies into the buffer
                self.his_policy_list.append(copy.deepcopy(self.policy))
                self.his_policy_list = self.his_policy_list[-10:]

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list),
                   'Loss/p_reg_loss': np.mean(p_reg_losses),
                   'Stats/policy_churn': np.mean(p_churn_amounts),
                   'Stats/value_churn': np.mean(v_churn_amounts),
                   }
        return summary