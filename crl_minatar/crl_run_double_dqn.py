################################################################################################################
# Authors (modified by):                                                                                                     #
# Hongyao Tang (tanghyyy@gmail.com)                                                                            #
#                                                                                                              #
# python run_random_agent.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################

import os, time, argparse, random
import numpy as np
from minatar import Environment
import torch
from agents.double_dqn import DoubleDQN
import wandb


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_env, eval_episodes=10, env_name=None):
    avg_return = 0.
    avg_step = 0.
    state_shape = [10, 10, 10]
    false_state_pad = np.zeros(shape=state_shape).astype(bool)
    for _ in range(eval_episodes):
        eval_env.reset()
        s = eval_env.state()
        s = np.concatenate([s, false_state_pad], axis=-1)[:, :, :10]
        done = False
        t = 0
        ep_reward = 0
        while not done:
            action = policy.select_action(s, is_greedy=True)
            reward, done = eval_env.act(action)
            s = eval_env.state()
            s = np.concatenate([s, false_state_pad], axis=-1)[:, :, :10]
            avg_return += reward
            avg_step += 1

            ep_reward += reward
            t += 1

            # BugFixed for Seaquest never done case
            if env_name is not None and env_name == 'seaquest':
                if t >= 1000 and ep_reward < 0.1:
                    break

    avg_return /= eval_episodes
    avg_step /= eval_episodes

    print("Evaluation over %d episodes - Score: %f, Steps: %f" % (eval_episodes, avg_return, avg_step))
    return avg_return, avg_step


def get_args_from_parser():
    parser = argparse.ArgumentParser()

    # breakout, freeway, asterix, seaquest, space_invaders
    parser.add_argument("--env", type=str, default='minatar3', help='env')
    parser.add_argument("--seed", type=int, default=1, help='random_seed')
    parser.add_argument("--max-step", type=int, default=2000, help='total_steps(k)')
    parser.add_argument("--cluster", type=int, default=0, help='use cluster or not')
    parser.add_argument("--gpu-no", type=str, default='1', help='gpu_no')

    parser.add_argument("--lr", type=float, default=0.0003, help='learning_rate')
    parser.add_argument("--ti", type=int, default=1, help='train_interval')
    parser.add_argument("--hard-replacement-interval", type=int, default=1000, help='hard replacement interval')

    parser.add_argument("--eval-interval", type=int, default=50000, help='number of steps per evaluation point')
    parser.add_argument("--is-save-data", action="store_true", help='is_save_data')

    # NOTE - hyperparams for policy churn investigation
    parser.add_argument("--invest", default=False, action='store_true')
    parser.add_argument('--invest_window_size', default=10, type=int)
    parser.add_argument('--invest_interval', default=2, type=int)

    return parser.parse_args()


def run_exp(args):
    #####################  hyper parameters  ####################

    MAX_TOTAL_STEPS = 1000 * args.max_step
    INIT_RANDOM_STEPS = 10000

    # Init wandb logger
    # Default hyperparam (partially) from MinAtar Paper
    hyperparam_config = dict(
        alg_name='double_dqn',
        total_max_steps=MAX_TOTAL_STEPS,
        init_random_steps=INIT_RANDOM_STEPS,
        batch_size=32,
        memory_size=500000,
        init_epsilon=1.0,
        end_epsilon=0.1,
        epsilon_decay_steps=500000,
    )
    hyperparam_config.update(vars(args))

    run_name = f"{hyperparam_config['env']}_{hyperparam_config['alg_name']}" \
               f"_invest_w{args.invest_window_size}i{args.invest_interval}" \
               f"_{hyperparam_config['seed']}_t{int(time.time())}"
    # offline mode is safer and more reliable for wandb, yet need manual sync
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        name=run_name,
        project="crl_minatar",
        config=hyperparam_config,
        save_code=True,
    )

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # - cpu occupation num
    cpu_num = 2
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    # Set device
    if args.cluster == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu': print('- Note cpu is being used now.')

    # NOTE - env list
    env_list_dict = {
        'minatar3': ['space_invaders', 'asterix', 'seaquest',],
        'minatar_saf': ['space_invaders', 'asterix', 'freeway',],
    }
    env_list = env_list_dict[args.env]

    agent = None
    total_global_step_count = 0
    state_shape = [10, 10, 10]
    false_state_pad = np.zeros(shape=state_shape).astype(bool)
    for env_name in env_list:
        print('=============================')
        print('Running for env: ', env_name)
        MAX_TOTAL_STEPS = 1000 * args.max_step

        # Init Env
        env = Environment(env_name)
        env.seed(args.seed)
        # Get channels and number of actions specific to each game
        # state_shape = env.state_shape()
        num_actions = env.num_actions()

        env4eval = Environment(env_name)
        env4eval.seed(args.seed * 1234)

        print('-- Env:', env_name)
        print('-- State space shape:', state_shape)
        print('-- Num action:', num_actions)
        print('-- Seed:', args.seed)
        print('-- Configurations:', vars(args))

        # Init RL agent
        if agent is None:
            print('- Init agent')
            agent = DoubleDQN(state_shape=state_shape, num_actions=num_actions, device=device,
                              learning_rate=args.lr, hard_replacement_interval=args.hard_replacement_interval,
                              invest=args.invest,
                              invest_window_size=args.invest_window_size, invest_interval=args.invest_interval)

        return_his, step_his = [], []
        q_loss_his, avg_q_loss_his = [], []
        evaluation_return_list, evaluation_step_list = [], []

        global_step_count = 0
        ep_num = 0

        # Initial evaluation
        print("---------------------------------------")
        print('- Evaluation at Step', global_step_count, '/', MAX_TOTAL_STEPS, f' ({total_global_step_count})')
        eval_return, eval_step = evaluate_policy(agent, eval_env=env4eval, eval_episodes=10, env_name=env_name)
        evaluation_return_list.append(eval_return)
        evaluation_step_list.append(eval_step)
        wandb.log({'eval_return': eval_return}, step=total_global_step_count)
        wandb.log({'eval_step': eval_step}, step=total_global_step_count)
        print("---------------------------------------")

        # Training
        run_start_time = time.time()
        while global_step_count < MAX_TOTAL_STEPS:
            env.reset()
            s = env.state()
            # NOTE - pad the state
            s = np.concatenate([s, false_state_pad], axis=-1)[:, :, :10]
            done = False

            ep_reward = 0
            ep_step_count = 0

            while (not done) and global_step_count < MAX_TOTAL_STEPS:
                # Interact with env
                if global_step_count < INIT_RANDOM_STEPS:
                    a = np.random.choice([i for i in range(num_actions)])
                else:
                    a = agent.select_action(s)
                    agent.epsilon_decay()

                r, done = env.act(a)
                s_ = env.state()
                # NOTE - pad the state
                s_ = np.concatenate([s_, false_state_pad], axis=-1)[:, :, :10]

                agent.store_experience(s, a, r, s_, int(done))

                if global_step_count >= INIT_RANDOM_STEPS:
                    loss = agent.train(step=global_step_count + 1)
                    q_loss_his.append(loss)

                if global_step_count % 10000 == 0:
                    wandb.log({"SPS": int(global_step_count / (time.time() - run_start_time))}, step=total_global_step_count)

                ep_step_count += 1
                global_step_count += 1
                total_global_step_count += 1

                s = s_
                ep_reward += r

                if global_step_count % args.eval_interval == 0:
                    print("---------------------------------------")
                    print('- Evaluation at Step', global_step_count, '/', MAX_TOTAL_STEPS, f' ({total_global_step_count})')
                    eval_return, eval_step = evaluate_policy(agent, eval_env=env4eval, eval_episodes=10, env_name=env_name)
                    evaluation_return_list.append(eval_return)
                    evaluation_step_list.append(eval_step)
                    wandb.log({'eval_return': eval_return}, step=total_global_step_count)
                    wandb.log({'eval_step': eval_step}, step=total_global_step_count)
                    print("---------------------------------------")

                # BugFixed for Seaquest never done case
                if env_name == 'seaquest':
                    if ep_step_count >= 1000 and ep_reward < 0.1:
                        break

            ep_num += 1
            return_his.append(ep_reward)
            step_his.append(ep_step_count)

            if ep_num % 10 == 0:
                avg_ep_return = sum(return_his[-10:]) / len(return_his[-10:])
                avg_ep_step = sum(step_his[-10:]) / len(step_his[-10:])
                avg_q_loss = sum(q_loss_his[-100:]) / len(q_loss_his[-100:]) if global_step_count >= INIT_RANDOM_STEPS else 0
                avg_q_loss_his.append(avg_q_loss)
                print('- Steps:', global_step_count, '/', MAX_TOTAL_STEPS, f' ({total_global_step_count})',
                      'Ep:', ep_num, ', return:', avg_ep_return, 'ep_steps:', avg_ep_step, 'avg_q_loss:', avg_q_loss)
                wandb.log({'training_return': avg_ep_return}, step=total_global_step_count)
                wandb.log({'training_step': avg_ep_step}, step=total_global_step_count)
                wandb.log({'q_loss': avg_q_loss}, step=total_global_step_count)

    if args.is_save_data:
        print('=========================')
        print('- Saving data.')

        save_folder_path = './results/crl_double_dqn'
        save_folder_path += '_lr' + str(args.lr)
        save_folder_path += '_hrfreq' + str(args.hard_replacement_interval)
        save_folder_path += '_ti' + str(args.ti)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        file_index = save_folder_path + args.env
        file_index += '_h' + str(args.max_step)
        file_index += '_s' + str(args.seed)
        np.savez_compressed(file_index,
                            q_loss=avg_q_loss_his,
                            eval_return=eval_return,
                            eval_step=eval_step,
                            config=vars(args),
                            )
        # save model
        # agent.save()

        print('- Data saved.')
        print('-------------------------')


if __name__ == '__main__':

    t1 = time.time()
    arguments = get_args_from_parser()
    run_exp(args=arguments)
    print('Running time: ', time.time() - t1)
