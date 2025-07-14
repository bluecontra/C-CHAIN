# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import copy
from utils.basic_utils import make_env, AgentReLU, eval_policy


@dataclass
class Args:
    scale_up_ratio: int = 1
    sparsity_level: float = 0.0
    target_rel_loss_scale: float = 0.05

    # Algorithm specific arguments
    alg: str = 'ppo_relu_pcra'
    """the name of designated algorithm"""
    env_id: str = "quadruped"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    eval_freq_timesteps: int = 20000
    """timestep to eval learned policy"""
    seed: int = 1
    """seed of the experiment"""

    gpu_no: str = '1'
    """designate the gpu with corresponding number to run the exp"""
    use_cluster: bool = False
    """0 for using normal physical workstation; 1 for using cluster"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    # anneal_lr: bool = True
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.alg}__{args.env_id}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            name=run_name,
            project='continual_rl_ppo_dmc',
            # entity='',
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Set device
    if not args.use_cluster:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu': print('- Note cpu is being used now.')

    agent = None
    # env setup
    env_name_dict = {
        'dog': ['dm_control/dog-stand-v0', 'dm_control/dog-walk-v0', 'dm_control/dog-run-v0', 'dm_control/dog-trot-v0',],
        'walker': ['dm_control/walker-stand-v0', 'dm_control/walker-walk-v0', 'dm_control/walker-run-v0'],
        'quadruped': ['dm_control/quadruped-walk-v0', 'dm_control/quadruped-run-v0', 'dm_control/quadruped-walk-v0'],
    }
    env_name_list = env_name_dict[args.env_id]

    life_step = 0
    for env_name in env_name_list:
        print('========================')
        print('Training on ', env_name)
        envs = gym.vector.SyncVectorEnv(
            [make_env(env_name, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        # NOTE - Eval env only resets once (this is somewhat special due to the implementation in CleanRL)
        eval_envs = gym.vector.SyncVectorEnv([make_env(env_name, 0, args.capture_video, run_name, args.gamma)])

        # agent init (only once)
        if agent is None:
            agent = AgentReLU(envs, scale_up_ratio=args.scale_up_ratio, sparsity_level=args.sparsity_level).to(device)
            optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate / np.sqrt(args.scale_up_ratio), eps=1e-5)

        norm_stats = [copy.deepcopy(e.obs_rms) for e in envs.envs]
        init_eval, init_horizon = eval_policy(agent, eval_envs, norm_stats, eval_episodes=10, device=device)
        print("---------------------------------------")
        print(f"T: {0}, Evaluation over {len(init_eval)} episodes. "
              f"Scores: {np.mean(init_eval):.3f}, Horizons: {np.mean(init_horizon):.3f}")
        print("---------------------------------------")
        # evaluations, horizons = [np.mean(init_eval)], [np.mean(init_horizon)]
        writer.add_scalar("charts/Eval", np.mean(init_eval), 0)
        writer.add_scalar("charts/Eval_Horizon", np.mean(init_horizon), 0)

        his_agent_list = [copy.deepcopy(agent)]
        # NOTE - init the auto-reg coef
        running_p_loss_list, running_reg_loss_list = [], []
        cur_reg_coef = 100.0

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()

        update_cnt = 0
        eval_cnt = 0
        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate / np.sqrt(args.scale_up_ratio)
                optimizer.param_groups[0]["lr"] = lrnow
            else:
                frac = 1.0

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                life_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, life_step={life_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], life_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], life_step)

            # Evaluate episode
            if (global_step // args.eval_freq_timesteps) > eval_cnt:
                eval_cnt += 1
                norm_stats = [copy.deepcopy(e.obs_rms) for e in envs.envs]
                cur_eval, cur_horizon = eval_policy(agent, eval_envs, norm_stats, eval_episodes=10, device=device)
                print("---------------------------------------")
                print(f"T: {global_step}, life_step={life_step}, Evaluation over {len(cur_eval)} episodes. "
                      f"Scores: {np.mean(cur_eval):.3f}, Horizons: {np.mean(cur_horizon):.3f}")
                print("---------------------------------------")
                # evaluations.append(np.mean(cur_eval))
                # horizons.append(np.mean(cur_horizon))
                writer.add_scalar("charts/Eval", np.mean(cur_eval), eval_cnt)
                writer.add_scalar("charts/Eval_Horizon", np.mean(cur_horizon), eval_cnt)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            b_ref_inds = np.arange(args.batch_size)
            b_reg_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                np.random.shuffle(b_ref_inds)
                np.random.shuffle(b_reg_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_ref_inds = b_ref_inds[start:end]
                    mb_reg_inds = b_reg_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    # NOTE - Churn reduction loss
                    if len(his_agent_list) <= 2:
                        reg_loss = 0
                    else:
                        reg_agent = his_agent_list[-2]
                        cur_action_means = agent.actor_mean(b_obs[mb_reg_inds])
                        with torch.no_grad():
                            reg_action_means = reg_agent.actor_mean(b_obs[mb_reg_inds])
                        reg_loss = ((cur_action_means - reg_action_means) ** 2).mean()
    
                    loss += reg_loss * cur_reg_coef
    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    # NOTE - add historical policies into the buffer
                    his_agent_list.append(copy.deepcopy(agent))
                    his_agent_list = his_agent_list[-10:]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            ref_agent = his_agent_list[-2]
            with torch.no_grad():
                cur_ref_action_means = agent.actor_mean(b_obs[mb_ref_inds])
                ref_action_means = ref_agent.actor_mean(b_obs[mb_ref_inds])
            policy_churn = ((cur_ref_action_means - ref_action_means) ** 2).mean()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], life_step)
            writer.add_scalar("charts/policy_churn", policy_churn.item(), life_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), life_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), life_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), life_step)
            writer.add_scalar("losses/reg_loss", reg_loss.item(), life_step)
            writer.add_scalar("charts/auto_coef", cur_reg_coef, life_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), life_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), life_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), life_step)
            writer.add_scalar("losses/explained_variance", explained_var, life_step)
            print("SPS:", int(life_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(life_step / (time.time() - start_time)), life_step)
    
            # NOTE - adjust reg coef according to the trend of entropy (we want the entropy decrease as fast as possible)
            running_p_loss_list.append(pg_loss.item())
            running_reg_loss_list.append(reg_loss.item())
            if iteration >= 50:
                running_p_loss = np.mean(np.abs(running_p_loss_list[-100:]))
                running_reg_loss = np.mean(running_reg_loss_list[-100:])
                cur_reg_coef = args.target_rel_loss_scale * running_p_loss / (running_reg_loss + 1e-8) * frac

        envs.close()

    writer.close()
