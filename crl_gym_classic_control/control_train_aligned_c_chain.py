"""
modified from https://github.com/ComputationalRobotics/TRAC/blob/main/main.ipynb
"""
import os, time
import argparse
import gym
import random
import copy

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from trac_optimizer import start_trac

from networks import PolicyNetwork, ValueNetwork
from basic_utils import Episode, History, train_combined_networks_with_chain


def get_peturbations(env_name, num_level, perturb_range):
    env = gym.make(env_name, render_mode="rgb_array")
    observation = env.reset()[0]
    random_perturbations = [
        np.random.normal(0, perturb_range, observation.shape) for _ in range(num_level)
    ]
    # make the first random perturbation zero
    random_perturbations[0] = np.zeros(observation.shape)
    return random_perturbations


if __name__ == '__main__':
    start_time = time.time()
    # - config setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Acrobot-v1', help='environment ID, Acrobot-v1, CartPole-v1, LunarLander-v2')
    parser.add_argument('--obs_perturb_range', type=float, default=2, help='perturbation range')
    parser.add_argument('--level_num', type=int, default=10, help='number of levels')
    parser.add_argument('--level_switch', type=int, default=200, help='number of iterations per level_switch')
    parser.add_argument('--seed', type=int, default=0, help='Random generator seed')
    parser.add_argument('--gpu_no', type=str, default='0', help='gpu no.')
    parser.add_argument("--use_cluster", default=False, action='store_true')
    
    parser.add_argument('--alg', type=str, default='ppo_cchain', help='algorithm name')
    parser.add_argument('--target_p_rel_loss_scale', type=float, default=10000, help='the target policy relative scale for chain')
    parser.add_argument('--opt', type=str, default='base', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--scale_up_ratio', type=int, default=1, help='scale up ratio (via widening)')
    parser.add_argument('--act_f', type=str, default='l_relu', help='activation function')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--iteration_steps', type=int, default=800, help='interaction step num per iteration')
    parser.add_argument('--epoch', type=int, default=5, help='training epoch')
    args = parser.parse_args()

    max_iterations = args.level_num * args.level_switch
    max_timesteps = 400
    state_scale = 1.0
    reward_scale = 20.0

    # - setup running env
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Set device
    # print('- visible device:', os.environ["CUDA_VISIBLE_DEVICES"])
    print('- cuda available:', torch.cuda.is_available())
    if not args.use_cluster:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('- GPU/CPU device:', device)

    # - set logger
    run_name = f'{args.env_name}l{args.level_num}s{args.level_switch}p{args.obs_perturb_range}_{args.opt}_s{args.seed}'
    import wandb
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        name=run_name,
        project='crl_ppo_control_aligned',
        entity='your_entity',
        config=vars(args),
        save_code=True,
    )

    # - generate perturbation
    random_perturbations = get_peturbations(args.env_name, args.level_num, args.obs_perturb_range)

    # - setup env
    env = gym.make(args.env_name, render_mode="rgb_array")
    observation = env.reset()[0]
    n_actions = env.action_space.n
    feature_dim = observation.size

    # - init agent
    value_model = ValueNetwork(in_dim=feature_dim, activation=args.act_f,
                               scale_up_ratio=args.scale_up_ratio, device=device).to(device)
    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions, activation=args.act_f,
                                 scale_up_ratio=args.scale_up_ratio, device=device).to(device)

    history_policy_list = [copy.deepcopy(policy_model)]
    history_value_list = [copy.deepcopy(value_model)]

    # - init alg
    if args.opt == "TRAC":
        trac_combined_optimizer = start_trac(log_file=f'trac_logs/{run_name}', Base=optim.Adam)(
            [
                {"params": policy_model.parameters(), "lr": args.lr},
                {"params": value_model.parameters(), "lr": args.lr},
            ]
        )
        combined_optimizer = trac_combined_optimizer
        print("USING TRAC.")
    elif args.opt == "base":
        base_combined_optimizer = torch.optim.Adam(
            [
                {"params": policy_model.parameters(), "lr": args.lr},
                {"params": value_model.parameters(), "lr": args.lr},
            ]
        )
        combined_optimizer = base_combined_optimizer
    else:
        print('- ERROR: Unknown optimizer.')
        raise NotImplementedError

    history = History()
    level = 0
    global_step_count = 0

    for ite in range(max_iterations):
        # Switch perturbation level
        if ite % args.level_switch == 0:
            random_perturbation = random_perturbations[level]
            level += 1
            # NOTE - reset periordic stats
            chain_p_reg_coef = 1 if args.target_p_rel_loss_scale > 0 else 0
            running_p_loss_list, running_v_loss_list = [], []
            running_p_reg_loss_list = []

        episodes_reward = []
        episodes_step = []
        ite_timestep = 0
        while ite_timestep < args.iteration_steps:
            observation = env.reset()[0]
            observation += random_perturbation
            episode = Episode()

            for timestep in range(max_timesteps):
                action, log_probability = policy_model.sample_action(observation / state_scale)
                value = value_model.state_value(observation / state_scale)

                new_observation, reward, done, _, _ = env.step(action)
                new_observation += random_perturbation

                episode.append(
                    observation=observation / state_scale,
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=reward_scale,
                )

                observation = new_observation
                global_step_count += 1
                ite_timestep += 1

                if done:
                    episode.end_episode(last_value=0)
                    break

                if timestep == max_timesteps - 1 or ite_timestep == args.iteration_steps:
                    value = value_model.state_value(observation / state_scale)
                    episode.end_episode(last_value=value)
                    break

            episodes_reward.append(reward_scale * np.sum(episode.rewards))
            history.add_episode(episode)
            episodes_step.append(timestep + 1)

        mean_rewards = np.mean(episodes_reward)
        mean_steps = np.mean(episodes_step)

        # NOTE - training stage
        history.build_dataset()
        data_loader = DataLoader(history, batch_size=args.batch_size, shuffle=True)
        reg_data_loader = DataLoader(history, batch_size=args.batch_size, shuffle=True)
        ref_data_loader = DataLoader(history, batch_size=args.batch_size, shuffle=True)
        train_stats_dict, history_policy_list, history_value_list = \
            train_combined_networks_with_chain(policy_model, value_model, combined_optimizer,
                                               data_loader, reg_data_loader, ref_data_loader,
                                               history_policy_list, history_value_list,
                                               p_reg_coef=chain_p_reg_coef,
                                               epochs=args.epoch, device=device)

        # NOTE - adjust reg coef according to the relative scale of losses
        running_p_loss_list.append(train_stats_dict['p_loss'])
        running_v_loss_list.append(train_stats_dict['v_loss'])
        running_p_reg_loss_list.append(train_stats_dict['p_reg_loss'])
        if ite - (level - 1) * args.level_switch >= 10:
            if args.target_p_rel_loss_scale > 0:
                running_p_loss = np.mean(np.abs(running_p_loss_list[-50:]))
                running_p_reg_loss = np.mean(running_p_reg_loss_list[-50:])
                chain_p_reg_coef = max(args.target_p_rel_loss_scale * running_p_loss / (running_p_reg_loss + 1e-8), 1)

        print(f'- Iteration num: {ite + 1}, Avg Return: {mean_rewards}, Avg Step: {mean_steps},'
              f" P Loss: {train_stats_dict['p_loss']:.2f}, V Loss: {train_stats_dict['v_loss']:.2f},"
              f" P_reg Loss: {train_stats_dict['p_reg_loss']:.2f}")
        # wandb.log({'global_timestep': global_step_count,
        #            'itr_mean_return': mean_rewards, 'itr_mean_step': mean_steps}, step=ite + 1)
        # NOTE - log training stats
        wandb.log({'global_timestep': global_step_count,
                   'itr_mean_return': mean_rewards, 'itr_mean_step': mean_steps,
                   'stats/auto_p_reg_coef': chain_p_reg_coef,
                   'stats/policy_churn': train_stats_dict['policy_churn'],
                   'stats/value_churn': train_stats_dict['value_churn'],
                   'stats/p_reg_loss': train_stats_dict['p_reg_loss'],
                   'stats/policy_loss': train_stats_dict['p_loss'],
                   'stats/value_loss': train_stats_dict['v_loss']},
                  step=ite + 1)
        history.free_memory()

    print('========================================')
    print('Exp done.')
    print('Running time: ', time.time() - start_time)










