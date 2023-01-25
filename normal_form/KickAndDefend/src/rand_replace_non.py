import os, sys
sys.path.append('..')
import numpy as np
import gym, argparse
import timeit
import gym_compete
from rollout import rollout
import tensorflow as tf
from agent_fidelity import make_zoo_agent,make_adv_agent
from gym import spaces
# Setup env, load the target agent, and collect the trajectories.


tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1)
sess = tf.Session(config=tf_config)
sess.__enter__()
sess.run(tf.variables_initializer(tf.global_variables()))


env_name = 'multicomp/KickAndDefend-v0'
victim_agent_path = '/home/zxc5262/rl_adv_valuediff/mujoco/multiagent-competition/agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl'
adv_agent_path = '/home/zxc5262/rl_adv_valuediff/mujoco/adv-agent/ucb/you/model.npy'
mask_agent_path = '/home/zxc5262/rl_adv_valuediff/selfplay_kicker/agent-zoo/KickAndDefend-v0_0_MLP_MLP_1_const_-1_const_0_const_False/20220203_004127-0/checkpoints/000019906560/model.pkl'
adv_ismlp = True
adv_obs_normpath = '/home/zxc5262/rl_adv_valuediff/mujoco/adv-agent/ucb/you/obs_rms.pkl'
mask_obs_normpath = '/home/zxc5262/rl_adv_valuediff/selfplay_kicker/agent-zoo/KickAndDefend-v0_0_MLP_MLP_1_const_-1_const_0_const_False/20220203_004127-0/checkpoints/000019906560/obs_rms.pkl'
mask_action_space = spaces.Discrete(2)
# Load agent, build environment, and play an episode.
env = gym.make(env_name)
env.seed(1)
# np.random.seed(100)
victim_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=1, version=1)
adv_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=2, version=1, scope='adv_agent')
with tf.variable_scope("mask_agent", reuse=False):
    mask_agent = make_adv_agent(env.observation_space.spaces[1], mask_action_space,1,mask_agent_path,adv_ismlp,mask_obs_normpath, name='mask_agent')


traj_path = 'trajs/' + env_name.split('/')[1]
#traj_path = 'trajs/Pong-v0.npz'
num_traj = 200
max_ep_len = 200
win_count = 0
valid_num = num_traj
replay_rewards = []

critical_steps_starts = np.loadtxt("./recording/non_critical_steps_starts.out")
critical_steps_ends = np.loadtxt("./recording/non_critical_steps_ends.out")

original_rewards = np.loadtxt("./recording/reward_record.out")

for i in range(num_traj):
    print("Process traj:", i)
    if original_rewards[i]==0:
        valid_num -= 1
        replay_rewards.append(0)
        continue
    vic_action_sequence_path = "./recording/vic_act_seq_" + str(i) + ".out"
    vic_recorded_actions = np.loadtxt(vic_action_sequence_path)

    adv_action_sequence_path = "./recording/adv_act_seq_" + str(i) + ".out"
    adv_recorded_actions = np.loadtxt(adv_action_sequence_path)

    iteration_ends_path = "./recording/eps_len_" + str(i) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    critical_frames_start = critical_steps_starts[i]
    critical_frames_end = critical_steps_ends[i]

    random_replacement_steps = critical_frames_end - critical_frames_start

    start_range = int(iteration_ends - random_replacement_steps)

    rand_frames_start = np.random.choice(start_range)

    rand_frames_end = rand_frames_start + random_replacement_steps

    env.seed(i)
    observation = env.reset()
    victim_agent.reset()
    adv_agent.reset()
    episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
    
    while not done and episode_length < max_ep_len:
        
        actions = []
        if episode_length < rand_frames_start:
            clip_adv_action = np.clip(adv_recorded_actions[episode_length], env.action_space[0].low, env.action_space[0].high)
            clip_vic_action = np.clip(vic_recorded_actions[episode_length], env.action_space[0].low, env.action_space[0].high)
            actions.append(clip_vic_action)
            actions.append(clip_adv_action)
        
        elif episode_length <= rand_frames_end:
            for id, obs in enumerate(observation):
                if id==0:
                   act = victim_agent.act(observation=obs)
                   clipped_actions = np.clip(act, env.action_space[0].low, env.action_space[0].high)
                else:
                    act = adv_agent.act(observation=obs)
                    act = act + np.random.rand(act.shape[0]) * 3 -1
                    clipped_actions = np.clip(act, env.action_space[0].low, env.action_space[0].high)
                actions.append(clipped_actions)

        else:
            for id, obs in enumerate(observation):
                if id==0:
                   act = victim_agent.act(observation=obs)
                   clipped_actions = np.clip(act, env.action_space[0].low, env.action_space[0].high)
                else:
                    act = adv_agent.act(observation=obs)
                    '''mask action'''
                    # mask_act, _ = mask_agent.act(observation=obs)
                    # mask_act = mask_act[0]
                    # if mask_act == 1:
                    #     act = act + np.random.rand(act.shape[0]) * 3 -1
                    '''mask action finish'''
                    clipped_actions = np.clip(act, env.action_space[0].low, env.action_space[0].high)
                actions.append(clipped_actions)


        actions = tuple(actions)
        observation, rewards, done, infos = env.step(actions)
        episode_length += 1


    if infos[1]['reward_remaining'] > infos[0]['reward_remaining']:
        replay_rewards.append(1)
        win_count += 1
    elif infos[0]['reward_remaining'] == infos[1]['reward_remaining']:
        replay_rewards.append(0)
        valid_num -= 1
    else:
        replay_rewards.append(-1)

np.savetxt("./recording/replay_rand_reward_record.out", replay_rewards)
print("The winning rate is: ", win_count/ valid_num)
