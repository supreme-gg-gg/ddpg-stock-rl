"""
PyTorch implementation of DDPG agent.
Setup up logging using tensorboard.
Can easily be modified to use wandb if you prefer.
"""

import os
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard integration

from .replay_buffer import ReplayBuffer
from ..base import BaseAgent
from .actor import StockActor
from .critic import StockCritic

class DDPGAgent(BaseAgent):
    def __init__(self, env, actor: StockActor, critic: StockCritic, gamma, actor_noise, 
                 obs_normalizer=None, action_processor=None,
                 config_file='config/default.json',
                 model_save_path='weights/ddpg/ddpg.pt', summary_path='results/ddpg/'):
        
        with open(config_file, "r") as f:
            self.config = json.load(f)
        np.random.seed(self.config['seed'])
        if env:
            env.seed(self.config['seed'])
        self.env = env
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
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
        """
        Saves model. Ideally should take in path as well to handle checkpoints but either way works.
        """
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }
        torch.save(checkpoint, self.model_save_path)
        print("Model saved in %s" % self.model_save_path)
    
    def load_model(self):
        """Load model either from checkpoint or for training."""
        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("Model loaded from %s" % self.model_save_path)
    
    def train(self, verbose=True, more_verbose=False):
        """
        During training, checkpoints are saved every 100 episodes.
        You can load the model from any checkpoint and restart training.
        When training is complete, the final model is saved.
        NOTE: num episode is defaulted to 500 for stock
        """
        num_episodes = self.config['episode']
        max_steps = self.config['max step']
        batch_size = self.config['batch size']
        checkpoint_freq = 100  # Save every 100 episodes
        np.random.seed(self.config['seed'])
        
        for ep in range(num_episodes):
            observation, _ = self.env.reset()
            if self.obs_normalizer:
                observation = self.obs_normalizer(observation)

            # Track episode statistics
            ep_reward = 0
            ep_max_q = 0
            ep_actor_loss = 0
            ep_critic_loss = 0
            
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
                        next_action = self.actor.predict_target(s2_batch)
                        target_q = self.critic.predict_target(s2_batch, next_action)
                        y = r_batch + self.gamma * target_q * done_batch
                    
                    current_q, critic_loss = self.critic.train_step(s_batch, a_batch, y)
                    _, actor_loss = self.actor.train_step(s_batch, self.critic)
                    
                    self.actor.update_target_network()
                    self.critic.update_target_network()
                    
                    ep_max_q += current_q.max().item()
                    ep_actor_loss += actor_loss
                    ep_critic_loss += critic_loss
                
                ep_reward += reward
                observation = next_obs

                if done or step == max_steps - 1:
                    avg_q = ep_max_q / (step + 1) if step > 0 else 0
                    avg_actor_loss = ep_actor_loss / (step + 1) if step > 0 else 0
                    avg_critic_loss = ep_critic_loss / (step + 1) if step > 0 else 0
                    if verbose:
                        print(f"Episode: {ep}, Reward: {ep_reward:.2f}, Avg Q: {avg_q:.4f}")
                    self.writer.add_scalar("Reward", ep_reward, ep)
                    self.writer.add_scalar("Avg_Q", avg_q, ep)
                    self.writer.add_scalar("Actor_Loss", avg_actor_loss, ep)
                    self.writer.add_scalar("Critic_Loss", avg_critic_loss, ep)
                    break
            
            # Save checkpoint periodically
            if (ep + 1) % checkpoint_freq == 0:
                checkpoint_path = self.model_save_path.replace('.pt', f'_ep{ep+1}.pt')
                checkpoint = {
                    'episode': ep + 1,
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict()
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at episode {ep+1}")
        
        self.save_model()  # Save final model
        self.writer.flush()
        print("Training completed.")
    
    def predict(self, observation):
        """
        Predict action given observation.
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        obs_tensor = torch.tensor(np.expand_dims(observation, axis=0), dtype=torch.float32, device=self.device)
        action = self.actor.predict(obs_tensor).squeeze(0).cpu().detach().numpy()
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        """
        Predict action given single observation.
        NOTE: I believe the original predict single is deprecated it just uses the regular predict function.
        """
        return self.predict(observation)
