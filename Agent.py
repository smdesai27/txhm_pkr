import random
from collections import deque
import numpy as np # type:ignore
import torch # type:ignore
import torch.nn as nn # type:ignore
import torch.optim as optim # type:ignore
import torch.nn.functional as F # type:ignore
from torch.distributions import Categorical # type:ignore

class PokerPolicyNet(nn.Module):
    """
    A deep q-network for playing poker
    process two seperate inputs: a card tensor and an action history tensor
    """
    action_space_size = 5

    def __init__(self, num_players:int=6, max_action_channels: int = None):
        super(PokerPolicyNet, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
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

        # Create combined head 
        # concatinates both card and action features
        self.combined_fc = nn.Sequential(
            nn.Linear(2 * 256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        # Output head
        # Actor Head : outputs action prob. --> policy
        self.actor_head = nn.Linear(512, PokerPolicyNet.action_space_size)

        # Critic head: outputs single state value
        self.critic_head = nn.Linear(512, 1)

    def forward(self, card_tensor, action_tensor):
        # Process each tensor in its respective tower
        card_features = self.card_tower(card_tensor)
        action_features = self.action_tower(action_tensor)

        # Concatenate the feature vectors from both towers
        combined_features = torch.cat((card_features, action_features), dim=1)

        # pass thru fully connected layer
        shared_ouput = self.combined_fc(combined_features)

        #output from each head
        action_logits = self.actor_head(shared_ouput)
        state_value = self.critic_head(shared_ouput)

        return action_logits, state_value
    
class Agent:
    """
    Agent class with PPO, GAE, and proper hyperparameter management.
    """
    def __init__(self, player_id: int, learning_rate: float = 0.001, 
                 discount_factor: float = 0.99, ppo_clip: float = 0.2, 
                 delta1: float = 2.2, delta2: float = 1.8, delta3: float = 1.0, 
                 gae_lambda: float = 0.95, ppo_epochs: int = 5, game = None, 
                 num_players:int=6, max_action_channels: int = None):
        
        self.game = game
        
        self.player_id = player_id
        self.gamma = discount_factor
        self.ppo_clip = ppo_clip
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the Deep Q-Network and move it to the selected device
        self.policy_net = PokerPolicyNet(num_players=num_players, max_action_channels=max_action_channels).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Memory to store trajectories for PPO updates
        self.memory = []
        self.rewards_per_episode = []

    def select_action(self, action_tensor: np.array, card_tensor: np.array, legal_actions: list) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Selects an action using an epsilon-greedy strategy with the DQN.
        """
        self.policy_net.eval()
        with torch.no_grad():

            card_tensor_t = torch.tensor(card_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor_t = torch.tensor(action_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)

            action_logits, state_value = self.policy_net(card_tensor_t, action_tensor_t)

            masked_logits = torch.full_like(action_logits, -float('inf'))

            masked_logits = torch.full_like(action_logits, -float('inf'))
            masked_logits[0, legal_actions] = action_logits[0, legal_actions]


            distribution = Categorical(logits=masked_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            return action.item(), log_prob, state_value.squeeze(0)

    
    def store_final_transition(self, transition, reward):
        self.memory.append({
                "obs": transition["obs"],
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
            return

        self.policy_net.train()

        # Unpack memory
        obs_batch = [m["obs"] for m in self.memory]
        actions = torch.tensor([m["action"] for m in self.memory], device=self.device).long()
        old_log_probs = torch.stack([m["log_prob"] for m in self.memory]).to(self.device)
        rewards = np.array([m["reward"] for m in self.memory])
        values = torch.stack([m["value"] for m in self.memory]).squeeze(1).to(self.device)

        # --- GAE Calculation ---
        advantages = np.zeros_like(rewards, dtype=np.float32)
        start_index = 0

        for episode_len in self.rewards_per_episode:
            last_advantage = 0.0 # Reset for each new episode
            end_index = start_index + episode_len
            
            # Slice the rewards and values for just this episode
            episode_rewards = rewards[start_index:end_index]
            episode_values = values[start_index:end_index]
            
            for t in reversed(range(episode_len)):
                next_value = episode_values[t+1] if t + 1 < episode_len else 0.0
                td_error = episode_rewards[t] + self.gamma * next_value - episode_values[t]
                
                # The index into the main 'advantages' array needs to be offset
                advantages[start_index + t] = td_error + self.gamma * self.gae_lambda * last_advantage
                last_advantage = advantages[start_index + t]
            
            start_index = end_index

        # Convert to tensor and normalize AFTER all episodes are processed
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)


        # --- PPO Epochs and Loss Calculation ---
        # Convert obs to tensors once to avoid repetition inside the loop
        card_tensors = torch.stack([torch.tensor(obs[0], dtype=torch.float32, device=self.device) for obs in obs_batch])
        action_tensors = torch.stack([torch.tensor(obs[1], dtype=torch.float32, device=self.device) for obs in obs_batch])

        actor_losses = []
        critic_losses = []
        
        for _ in range(self.ppo_epochs):
            # Compute current logits and values
            logits, state_values = self.policy_net(card_tensors, action_tensors)
            
            # Actor Loss
            probs = F.softmax(logits, dim=-1)
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            log_probs = torch.log(action_probs + 1e-10)
            
            # Calculate ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs)

            # Trinal-Clip PPO loss
            clip1 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            clip2 = torch.clamp(ratio, 1 - self.delta1, 1 + self.delta1) * advantages
            clip3 = torch.clamp(ratio, 1 - self.delta2, 1 + self.delta2) * advantages
            clip4 = torch.clamp(ratio, 1 - self.delta3, 1 + self.delta3) * advantages

            actor_loss = -torch.min(torch.min(torch.min(ratio * advantages, clip1), clip2), torch.min(clip3, clip4)).mean()

            # Critic Loss (target is discounted returns)
            # The .squeeze() and .detach() are to make sure tensor shapes align correctly
            returns = advantages + values.detach()
            critic_loss = F.mse_loss(state_values.squeeze(1), returns)

            # Total loss
            loss = actor_loss + critic_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        # Clear memory after all epochs
        self.prev_mem_len = len(self.memory)
        self.memory.clear()
        self.rewards_per_episode.clear()

        return np.mean(actor_losses), np.mean(critic_losses)

    def save_policy(self, file_path):
        """
        Saves the agent's policy (actor and critic models) to a file.
        :param file_path: The path to the file where the policy will be saved.
        """
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f'policy saved to {file_path}')

    def watch_playthrough(self, num_episodes=1):
        """
        Runs a play-through with the loaded policy and prints the game state.
        """
        print("--- Starting Play-Through with Loaded Policy ---")
        poker_agent = self.poker_agent

        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            self.game.reset_hand()
            
            # Set the agent to evaluation mode
            poker_agent.policy_net.eval()
            
            while not self.game.check_terminal():
                # Handle chance nodes (card deals)
                while self.game.check_chance_node():
                    self.game.random_chance_node()
                    print(f"Board cards: {self.game.get_board_cards()}")

                if self.game.check_terminal():
                    break
                
                current_player = self.game.get_current_player()
                card_tensor = self.game.get_card_tensor(current_player)
                action_tensor = self.game.get_action_tensor()
                legal_actions = self.game.legal_actions()

                # Get the action from the agent's policy without exploration
                action, _, _ = poker_agent.select_action(action_tensor, card_tensor, legal_actions)

                # Print the action and game state
                action_name = self.game.get_action_name(action)
                print(f"Player {current_player} takes action: {action_name}")
                
                # Apply the action to the game
                self.game.apply_action(action)
            
            # Print the final result of the hand
            rewards = self.game.get_rewards()
            print(f"\n--- Episode {episode + 1} Concluded ---")
            print(f"Final Pot Size: {self.game.get_pot()}")
            print("Final Player Rewards:")
            for player_id, reward in rewards.items():
                print(f"  Player {player_id}: {reward}")

    def load_policy(self, file_path):
        """
        Loads a pre-trained agent policy from a file.
        :param file_path: The path to the file where the policy is saved.
        """
        if not torch.cuda.is_available():
            print("CUDA not available, loading on CPU.")

        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Policy successfully loaded from {file_path}")
        except FileNotFoundError:
            print(f"Error: Policy file not found at {file_path}")
        except Exception as e:
            print(f"An error occurred while loading the policy: {e}")
    
    