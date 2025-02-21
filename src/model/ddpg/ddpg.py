"""
PyTorch implementation of DDPG agent.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard integration

from .replay_buffer import ReplayBuffer
from ..base import BaseAgent
from .actor import StockActor
from .critic import StockCritic

class DDPGAgent(BaseAgent):
    def __init__(self, env, actor: StockActor, critic: StockCritic, actor_noise, obs_normalizer=None, action_processor=None,
                 config_file='config/default.json',
                 model_save_path='weights/ddpg/ddpg.pt', summary_path='results/ddpg/'):
        with open(config_file) as f:
            self.config = json.load(f)
        np.random.seed(self.config['seed'])
        if env:
            env.seed(self.config['seed'])
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.model_save_path = model_save_path
        self.summary_path = summary_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # send networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.buffer = ReplayBuffer(self.config['buffer size'])
        self.writer = SummaryWriter(log_dir=self.summary_path)  # Initialize TensorBoard writer
    
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }
        torch.save(checkpoint, self.model_save_path)
        print("Model saved in %s" % self.model_save_path)
    
    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Model loaded from %s" % self.model_save_path)
    
    def train(self, verbose=True):
        num_episodes = self.config['episode']
        max_steps = self.config['max step']
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        np.random.seed(self.config['seed'])
        
        for ep in range(num_episodes):
            observation, _ = self.env.reset()
            if self.obs_normalizer:
                observation = self.obs_normalizer(observation)
            ep_reward = 0
            ep_max_q = 0
            
            for step in range(max_steps):
                obs_tensor = torch.tensor(np.expand_dims(observation, axis=0), dtype=torch.float32, device=self.device)
                action = self.actor.predict(obs_tensor).squeeze(0).cpu().detach().numpy()
                action += self.actor_noise()
                if self.action_processor:
                    action_taken = self.action_processor(action)
                else:
                    action_taken = action

                next_obs, reward, done, _ = self.env.step(action_taken)
                if self.obs_normalizer:
                    next_obs = self.obs_normalizer(next_obs)
                
                self.buffer.add(observation, action, reward, done, next_obs)
                
                if self.buffer.size() >= batch_size:
                    # NOTE: s2 is the next state, s is the current state
                    s_batch, a_batch, r_batch, done_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    s_batch = torch.tensor(s_batch, dtype=torch.float32, device=self.device)
                    a_batch = torch.tensor(a_batch, dtype=torch.float32, device=self.device)
                    r_batch = torch.tensor(r_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
                    done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
                    s2_batch = torch.tensor(s2_batch, dtype=torch.float32, device=self.device)
                    
                    # Compute target Q value using target networks.
                    with torch.inference_mode():
                        # Use the actor to take action
                        next_action = self.actor.predict_target(s2_batch)
                        # Use the critic to evaluate (Q-value)
                        target_q = self.critic.predict_target(s2_batch, next_action)
                        # Compute the target Q value (Bellman equation)
                        y = r_batch + gamma * target_q * (1 - done_batch)
                    
                    # Update critic: MSE loss.
                    current_q = self.critic.train_step(s_batch, a_batch, y)
                    
                    # Update actor using the sampled policy gradient.
                    # pass in the critic to compute the gradients
                    self.actor.train_step(s_batch, self.critic)
                    
                    # Update target networks.
                    self.actor.update_target_network()
                    self.critic.update_target_network()
                    
                    ep_max_q += current_q.max().item()
                
                ep_reward += reward
                observation = next_obs
                if done or step == max_steps - 1:
                    avg_q = ep_max_q / (step + 1) if step > 0 else 0
                    if verbose:
                        print(f"Episode: {ep}, Reward: {ep_reward:.2f}, Avg Q: {avg_q:.4f}")
                    # Log to TensorBoard
                    self.writer.add_scalar("Reward", ep_reward, ep)
                    self.writer.add_scalar("Avg_Q", avg_q, ep)
                    break
        
        self.save_model()
        self.writer.flush()  # Ensure logs are written
        print("Training completed.")
    
    def predict(self, observation):
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        obs_tensor = torch.tensor(np.expand_dims(observation, axis=0), dtype=torch.float32, device=self.device)
        action = self.actor.predict(obs_tensor).squeeze(0).cpu().detach().numpy()
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        return self.predict(observation)
