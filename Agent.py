import random
import numpy as np
import torch
import torch.nn as nn #type:ignore
import torch.optim as optim #type:ignore
from torch.distributions import Categorical
import torch.nn.functional as F  #type:ignore
class PokerPolicyNet(nn.Module):
    """
    A deep q-network for playing poker
    process two seperate inputs: a card tensor and an action history tensor
    Now also includes stack features for better decision making
    """
    action_space_size = 5

    def __init__(self, num_players:int=6, max_action_channels: int = None):
        super(PokerPolicyNet, self).__init__()
        self.num_players = num_players
        self.actions_per_player = self.num_players * 2

        # Create tower 1 : Card feature extraction
        # This tower processes the (6, 4, 13) card tensor.
        self.card_tower = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels= 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 13, 256),
            nn.ReLU()
        )

        # Create tower 2 : Action feature extraction
        # 72, 8, 5
        # This tower processes the action tensor.
        flattened_size = 32 * (self.num_players + 2) * self.action_space_size
        self.action_tower = nn.Sequential(
            nn.Conv2d(in_channels=max_action_channels, out_channels=16, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU()
        )

        # added stick featuers
        # Processes [my_stack, opp_stack, pot_size, spr]
        self.stack_tower = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Create combined head 
        # concatinates card, action, and stack features
        self.combined_fc = nn.Sequential(
            nn.Linear(2 * 256 + 64, 512),  # Added 64 for stack features
            nn.ReLU(),
        )

        # Output head
        # Actor Head : outputs action prob. --> policy
        self.actor_head = nn.Linear(512, PokerPolicyNet.action_space_size)

        # Critic head: outputs single state value
        self.critic_head = nn.Linear(512, 1)

    def forward(self, card_tensor, action_tensor, stack_tensor=None):
        # Process each tensor in its respective tower
        card_features = self.card_tower(card_tensor)
        action_features = self.action_tower(action_tensor)

        # Process stack features if provided (backward compatible)
        if stack_tensor is not None:
            stack_features = self.stack_tower(stack_tensor)
            # Concatenate all feature vectors
            combined_features = torch.cat((card_features, action_features, stack_features), dim=1)
        else:
            # Fallback: use zero stack features for backward compatibility
            batch_size = card_features.shape[0]
            zero_stack_features = torch.zeros((batch_size, 64), device=card_features.device)
            combined_features = torch.cat((card_features, action_features, zero_stack_features), dim=1)

        # pass thru fully connected layer
        shared_ouput = self.combined_fc(combined_features)

        #output from each head
        action_logits = self.actor_head(shared_ouput)
        state_value = self.critic_head(shared_ouput)

        return action_logits, state_value


class Agent:
    """
    Agent class with PPO, GAE, and proper hyperparameter management.
    Includes fixes for training instability and exploration.
    """
    def __init__(self, player_id: int, learning_rate: float = .00003,
                 discount_factor: float = 0.99, ppo_clip: float = 0.2, 
                 delta1: float = 1.5, delta2: float = 1.2, delta3: float = 0.8, 
                 gae_lambda: float = 0.95, ppo_epochs: int = 5, game = None, 
                 num_players:int=6, max_action_channels: int = None,
                 epsilon_start=0.5, epsilon_decay=0.9995, epsilon_min=0.01):
        
        self.game = game
        self.player_id = player_id
        self.gamma = discount_factor
        self.ppo_clip = ppo_clip
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        
        # Keep track of the learning rate for loading policies
        self.learning_rate = learning_rate

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PokerPolicyNet(num_players=num_players, max_action_channels=max_action_channels).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)  # FIXED: Increased step_size

        self.memory = []
        self.rewards_per_episode = []

    def select_action(self, action_tensor: np.array, card_tensor: np.array, legal_actions: list, stack_tensor: np.array = None) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Selects an action using an epsilon-greedy strategy with weighted exploration.
        """
        if random.random() < self.epsilon:
            # --- FIXED: Weighted exploration favoring conservative actions ---
            # Weights: [Fold: 0.15, Call: 0.45, Raise1: 0.25, Raise2: 0.10, All-in: 0.05]
            action_weights = [0.15, 0.45, 0.25, 0.10, 0.05]
            available_weights = [action_weights[a] for a in legal_actions]
            normalized_weights = np.array(available_weights) / sum(available_weights)
            action = np.random.choice(legal_actions, p=normalized_weights)
            
            self.policy_net.eval()
            with torch.no_grad():
                card_tensor_t = torch.tensor(card_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_tensor_t = torch.tensor(action_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
                stack_tensor_t = torch.tensor(stack_tensor, dtype=torch.float32, device=self.device).unsqueeze(0) if stack_tensor is not None else None
                action_logits, state_value = self.policy_net(card_tensor_t, action_tensor_t, stack_tensor_t)
                
                masked_logits = torch.full_like(action_logits, -float('inf'))
                masked_logits[0, legal_actions] = action_logits[0, legal_actions]
                
                distribution = Categorical(logits=masked_logits)
                log_prob = distribution.log_prob(torch.tensor(action, device=self.device))
            return action, log_prob, state_value.squeeze(0)

        self.policy_net.eval()
        with torch.no_grad():
            card_tensor_t = torch.tensor(card_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor_t = torch.tensor(action_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
            stack_tensor_t = torch.tensor(stack_tensor, dtype=torch.float32, device=self.device).unsqueeze(0) if stack_tensor is not None else None

            action_logits, state_value = self.policy_net(card_tensor_t, action_tensor_t, stack_tensor_t)

            masked_logits = torch.full_like(action_logits, -float('inf'))
            masked_logits[0, legal_actions] = action_logits[0, legal_actions]

            distribution = Categorical(logits=masked_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            return action.item(), log_prob, state_value.squeeze(0)
    
    def store_final_transition(self, transition, reward):
        self.memory.append({
                "obs": transition["obs"],  # Now includes (card_tensor, action_tensor, stack_tensor)
                "action": transition["action"],
                "log_prob": transition["log_prob"].detach(),
                "reward": reward,
                "value": transition["value"].detach()
            })
    
    def add_episode_len(self, episode_len:int):
        self.rewards_per_episode.append(episode_len)

    def learn(self):
        """
        Updates the Actor-Critic network using the Trinal-Clip PPO loss.
        """
        if not self.memory:
            return 0.0, 0.0

        self.policy_net.train()

        obs_batch = [m["obs"] for m in self.memory]
        actions = torch.tensor([m["action"] for m in self.memory], device=self.device).long()
        old_log_probs = torch.stack([m["log_prob"] for m in self.memory]).to(self.device)
        rewards = np.array([m["reward"] for m in self.memory], dtype=np.float32)
        values = torch.stack([m["value"] for m in self.memory]).squeeze(1).to(self.device)

        # --- FIXED: Reward scaling instead of normalization to preserve magnitude ---
        # This preserves the difference between winning small pots and large all-in pots
        rewards = rewards / 10000.0  # Scale by typical stack size

        # --- GAE Calculation ---
        advantages = np.zeros_like(rewards, dtype=np.float32)
        start_index = 0
        for episode_len in self.rewards_per_episode:
            last_advantage = 0.0
            end_index = start_index + episode_len
            episode_rewards = rewards[start_index:end_index]
            episode_values = values[start_index:end_index]
            for t in reversed(range(episode_len)):
                next_value = episode_values[t+1] if t + 1 < episode_len else 0.0
                td_error = episode_rewards[t] + self.gamma * next_value - episode_values[t]
                advantages[start_index + t] = td_error + self.gamma * self.gae_lambda * last_advantage
                last_advantage = advantages[start_index + t]
            start_index = end_index
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # --- FIXED: Use clipping instead of normalization to preserve magnitude ---
        advantages = torch.clamp(advantages, -10.0, 10.0)

        # --- PPO Epochs and Loss Calculation ---
        card_tensors = torch.stack([torch.tensor(obs[0], dtype=torch.float32, device=self.device) for obs in obs_batch])
        action_tensors = torch.stack([torch.tensor(obs[1], dtype=torch.float32, device=self.device) for obs in obs_batch])
        # Handle stack tensors (may be None for backward compatibility)
        stack_tensors = None
        if len(obs_batch[0]) > 2 and obs_batch[0][2] is not None:
            stack_tensors = torch.stack([torch.tensor(obs[2], dtype=torch.float32, device=self.device) for obs in obs_batch])
        
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.ppo_epochs):
            logits, state_values = self.policy_net(card_tensors, action_tensors, stack_tensors)
            distribution = Categorical(logits=logits)
            entropy_bonus = distribution.entropy().mean()
            
            probs = F.softmax(logits, dim=-1)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            log_probs = torch.log(action_probs + 1e-10)
            
            ratio = torch.exp(log_probs - old_log_probs)

            clip1 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            clip2 = torch.clamp(ratio, 1 - self.delta1, 1 + self.delta1) * advantages
            clip3 = torch.clamp(ratio, 1 - self.delta2, 1 + self.delta2) * advantages
            clip4 = torch.clamp(ratio, 1 - self.delta3, 1 + self.delta3) * advantages
            actor_loss = -torch.min(torch.min(torch.min(ratio * advantages, clip1), clip2), torch.min(clip3, clip4)).mean()

            returns = advantages + values.detach()
            critic_loss = F.mse_loss(state_values.squeeze(1), returns)
            
            # --- FIXED: Reduced entropy bonus to prevent excessive exploration ---
            entropy_coefficient = 0.01  # Reduced from 0.1 to encourage more conservative play
            loss = actor_loss + critic_loss - entropy_coefficient * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)

            self.optimizer.step()
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        self.scheduler.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.prev_mem_len = len(self.memory)
        self.memory.clear()
        self.rewards_per_episode.clear()

        return np.mean(actor_losses), np.mean(critic_losses)

    def save_policy(self, file_path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, file_path)
        print(f'policy saved to {file_path}')

    def load_policy(self, file_path):
        """
        Loads a pre-trained agent policy and optimizer state from a file.
        Also forces the learning rate to the value specified in the constructor.
        """
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # --- NEW: Force the learning rate to its intended value ---
            # This is crucial for ensuring the correct LR is used when resuming.
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']

            print(f"Policy successfully loaded from {file_path}")

        except FileNotFoundError:
            print(f"Error: Policy file not found at {file_path}. Starting with a new, untrained policy.")
        except Exception as e:
            print(f"An error occurred while loading the policy: {e}. Starting with a new, untrained policy.")
        
    def load_state_dict(self, dict):
        self.policy_net.load_state_dict(dict)

    def save_policy_dict(self):
        return self.policy_net.state_dict()

