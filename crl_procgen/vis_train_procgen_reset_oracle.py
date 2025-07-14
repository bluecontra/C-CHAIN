# Adapted from https://github.com/joonleesky/train-procgen-pytorch

from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.policy import CategoricalPolicy
from common.model import ImpalaModel
from agents.ppo import PPO, environment_generator_procgen
from common import set_global_seeds, set_global_log_levels

import os, time, argparse
from procgen import ProcgenEnv
import random
import numpy as np
import torch
from pathlib import Path
import wandb
from trac_optimizer import start_trac


if __name__ == '__main__':
    start_time = time.time()
    # - config setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='ppo_reset_oracle', help='alg name')
    parser.add_argument('--exp_name', type=str, default='procgen', help='experiment name')
    parser.add_argument('--env_name', type=str, default='starpilot', help='environment ID')
    parser.add_argument('--level_steps', type=int, default=int(2e6), help='Number of steps per task')
    parser.add_argument('--max_levels', type=int, default=int(10), help='Number of distribution shifts')
    parser.add_argument('--env_seed', type=int, default=int(20), help='Environment')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')

    parser.add_argument('--opt', type=str, default='base', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--update_steps', type=int, default=int(1000), help='Number of steps per update')
    parser.add_argument('--storage_r', type=int, default=int(1), help='s1 or s2')
    parser.add_argument('--warmstart_step', type=int, default=int(0), help='warmstart step')

    parser.add_argument('--log_level', type=int, default=int(5), help='[10,20,30,40]')
    parser.add_argument('--log_freq', type=int, default=int(10), help='[10,100]')
    parser.add_argument('--gpu_no', type=str, default='0', help='gpu no.')
    parser.add_argument("--use_cluster", default=False, action='store_true')

    args = parser.parse_args()

    num_timesteps = args.level_steps * args.max_levels
    storage_r = True if args.storage_r == 1 else 0
    set_global_log_levels(args.log_level)

    # - setup running env
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_global_seeds(args.seed)

    # Set device
    if not args.use_cluster:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: " + str(device))

    # - set logger
    run_name = f'{args.env_name}_ml{args.max_levels}_ls{args.level_steps}_us{args.update_steps}_sr{args.storage_r}' \
               f'_alg{args.alg}_opt{args.opt}_warm{args.warmstart_step}_s{args.seed}'

    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        name=run_name,
        project='crl_ppo_procgen',
        entity='your_entity',
        config=vars(args),
        save_code=True,
    )

    # - setup env
    # torch.set_num_threads(1)
    torch.set_num_threads(4)
    if args.exp_name == 'procgen':
        env = ProcgenEnv(num_envs=1,
                         env_name=args.env_name,
                         start_level=0,
                         num_levels=1,
                         distribution_mode='hard')
        env = VecExtractDictObs(env, "rgb")
        env = VecNormalize(env, ob=False)
        env = TransposeFrame(env)
        env = ScaledFloatFrame(env)
    else:
        raise NotImplementedError

    observation_space = env.observation_space
    observation_shape = observation_space.shape
    in_channels = observation_shape[0]
    action_space = env.action_space

    # - setup logdir
    root = f'{args.exp_name}_logs/{args.env_name}_{args.level_steps}_{args.update_steps}_{args.storage_r}/'
    if args.warmstart_step:
        logdir = f'{root}{args.opt}_warmstarted_envseed{args.env_seed}_seed{args.seed}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
    else:
        logdir = f'{root}{args.opt}_envseed{args.env_seed}_seed{args.seed}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
    Path(logdir).mkdir(parents=True, exist_ok=True)
    optimizer_log_file_path = os.path.join(logdir, f"sinit_seed{args.seed}.txt")
    logger = Logger(1, logdir)

    if args.exp_name == 'procgen':
        env_generator = environment_generator_procgen(args.env_name, args.max_levels, seed=args.env_seed)
    else:
        raise NotImplementedError

    # - run training
    print('START TRAINING with ' + str(args.opt) + '...')
    # agent.train_seq(env_generator, optimizer_log_file_path=optimizer_log_file_path, exp_name=args.exp_name,
    #                 total_steps_per_level=args.level_steps, optimizer=args.opt, storage_r=storage_r,
    #                 warmstart=args.warmstart_step)

    steps_per_optimization = args.update_steps
    done = np.zeros(1)

    current_level = 0
    global_step = 0
    truncated = False
    for cur_env in env_generator:
        # NOTE - reinit agent for each env
        # - init policy
        if args.exp_name == 'procgen':
            model = ImpalaModel(in_channels=in_channels)
        else:
            raise NotImplementedError

        action_size = action_space.n
        policy = CategoricalPolicy(model, action_size)
        policy.to(device)

        hidden_state_dim = model.output_dim
        storage = Storage(observation_shape, hidden_state_dim, args.update_steps, 1, device)

        # - setup agent
        agent = PPO(env, policy, None, storage, device, 1, args.update_steps, learning_rate=args.lr)
        # agent = PPO(env, policy, logger, storage, device, 1, args.update_steps)
        hidden_state = np.zeros((1, storage.hidden_state_size))

        current_level += 1
        level_steps = 0
        level_iters = 0
        obs = cur_env.reset()
        while level_steps < args.level_steps:
            if global_step - (current_level - 1) * args.level_steps == args.warmstart_step:
                if args.opt == "TRAC":
                    optimizer = start_trac(log_file=optimizer_log_file_path, Base=torch.optim.Adam)(
                        policy.parameters(), lr=args.lr)
                else:
                    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
            # NOTE - Warning: manual reset ignored
            obs = cur_env.reset()
            policy.eval()
            for _ in range(steps_per_optimization):
                act, log_prob_act, value, next_hidden_state = agent.predict(obs, hidden_state, done)
                next_obs, rew, done, info = cur_env.step(act)

                storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state

                level_steps += 1
                global_step += 1
                if done or truncated:
                    obs = cur_env.reset()

            _, _, last_val, hidden_state = agent.predict(obs, hidden_state, done)
            storage.store_last(obs, hidden_state, last_val)
            storage.compute_estimates(agent.gamma, agent.lmbda, agent.use_gae, agent.normalize_adv)
            summary = agent.train(optimizer=optimizer)
            # self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = storage.fetch_log_data()
            logger.feed(rew_batch, done_batch)
            episode_stats = logger.get_episode_statistics()

            level_iters += 1
            if level_iters % args.log_freq == 0:
                wandb.log(summary, step=global_step)
                wandb.log(episode_stats, step=global_step)
                # logger.write_summary(summary)
                # logger.dump()
                print(f"- L:{current_level}, LS:{level_steps}, GS:{global_step},"
                      f" Mean Ep R:{episode_stats['Rewards/mean_episodes']:.2f},"
                      f" Avg Ep S:{episode_stats['Len/mean_episodes']}")
            if storage_r:
                storage.reset()
        if not storage_r:
            storage.reset()
        print("LEVEL SWITCHED")

    cur_env.close()
    
    

    print('========================================')
    print('Exp done.')
    print('Running time: ', time.time() - start_time)