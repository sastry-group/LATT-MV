from mujoco_env_only_kuka_ros import KukaTennisEnv
from gym.wrappers import FrameStack
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import rospy
# Function to create the environment
def make_env():
    return Monitor(KukaTennisEnv())

# def make_test_env():
#     return Monitor(KukaTennisEnv())

if __name__ == '__main__':
    rospy.init_node('kuka_joint_controller_training', anonymous=True)
    env = KukaTennisEnv()
    check_env(env)
    # Number of environments to run in parallel
    # num_envs = 256
    # env_list = [make_env for _ in range(num_envs)]

    # # Use SubprocVecEnv to parallelize the environments
    # env = DummyVecEnv(env_list)

    # Define custom architecture
    policy_kwargs = dict(
        net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],  # Two hidden layers with 64 units each for policy and value networks,
        activation_fn=nn.ELU  # Use ELU activation function
    )

    # eval_env = DummyVecEnv([make_test_env])
    # Set up evaluation callback
    eval_callback = EvalCallback(env, 
                                best_model_save_path='./logs/best_model/',
                                log_path='./logs/',
                                eval_freq=300,  # Evaluate every 10,000 steps
                                n_eval_episodes=3,  # Evaluate for 5 episodes
                                deterministic=True,  # Use deterministic actions during evaluation
                                render=True)
    # Initialize the PPO model
    model = PPO.load("logs/best_model/best_model",env)
    model.set_env(env)
    model.batch_size = 128
    # model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=3, 
    #         n_steps=50,   # Divide by num_envs to balance steps per environment
    #         batch_size=128,              # Minibatch size for PPO
    #         n_epochs=5,                # Number of epochs
    #         learning_rate=5e-4,         # Learning rate for PPO
    #         gamma=0.99,                 # Discount factor
    #         tensorboard_log="./ppo_kuka_parallel_tensorboard_ros/",
    #         device="cuda"
    #        )

    policy = model.policy

    # exit(0)
    
    
    # Train the PPO model
    model.learn(total_timesteps=5_000_000, callback=eval_callback)

    # Save the model
    model.save("ppo_kuka_parallel_mujoco_ros")

    # To load later
    # model = PPO.load("ppo_kuka_parallel_mujoco")
