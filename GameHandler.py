from GameWrapper import GameWrapper
from Agent import Agent
import random
import torch
import copy
from NemesisBots import NemesisBot, NitBot, AggressiveBlufferBot

class GameHandler:

    num_players = 2

    # k self play params -- designed for 2 players 
    K_OPPONENTS = 10
    SAVE_FREQUENCY = 1000
    INITIAL_ELO = 1200

    if num_players == 2:
        blind_config = "100 50"
        stack_config = "20000 20000"
    elif num_players == 6:
        blind_config = "100 50 0 0 0 0"
        stack_config = "20000 20000 20000 20000 20000 20000"
    
    # 6 player no limit hold em standard config 
    config = {
        "betting": "nolimit",
        "numPlayers": num_players,
        "numRounds": 4,
        "blind": blind_config,
        "firstPlayer": "2 1 1 1",
        "numSuits": 4,
        "numRanks": 13,
        "numBoardCards": "0 3 1 1",
        'numHoleCards' : 2,
        "stack": stack_config,
        "bettingAbstraction": "fchpa",
        "potSize": 0,
        "boardCards": "",
        "maxRaises": "255"
    }

    def __init__(self):
        self.agent_list = []
        self.game = GameWrapper(GameHandler.config)
        # Initialize a single agent that will play against itself
        max_action_channels = self.game.max_channels
        self.poker_agent = Agent(player_id=0, game = self.game, num_players=GameHandler.config["numPlayers"], 
                                 max_action_channels = max_action_channels)
        self.poker_agent_elo = self.INITIAL_ELO

        #IF REFINEING USE CALLING STATION NEMESIS AGENT
        self.nemesis_agent = NemesisBot(
            player_id=1,
            max_action_per_round=self.game.max_action_per_round
        )
        self.aggressive_bot = AggressiveBlufferBot(player_id=1, max_action_per_round=self.game.max_action_per_round)
        self.nit_bot = NitBot(player_id=1, max_action_per_round=self.game.max_action_per_round)

        #historical agent pool
        # entries as dict {id : iteration, state_dict: weights, elo: rating}
        self.historical_pool = []

        # add agent to pool
        self.historical_pool.append({
            'id': 'nemesis-calling-station',
            'state_dict': None, # No neural network
            'elo': 1600,
            'type': 'nemesis' # A flag to identify this special agent
        })
        self.historical_pool.append({
            'id': 'aggressive-bluffer', 'state_dict': None, 'elo': 1200, 'type': 'aggressive'
        })
        self.historical_pool.append({
            'id': 'nit-bot', 'state_dict': None, 'elo': 1200, 'type': 'nit'
        })

    
    def _add_agent_to_pool(self, iteration):
        """Clones the main agent's policy and adds it to the historical pool."""
        print(f"--- Snapshotting agent from iteration {iteration} into the pool. ---")
        # We use deepcopy to ensure the state_dict is a separate object
        new_historical_agent = {
            'id': iteration,
            'state_dict': copy.deepcopy(self.poker_agent.save_policy_dict()),
            'elo': self.poker_agent_elo,
            'type': 'historical' # Flag to identify standard agents
        }
        self.historical_pool.append(new_historical_agent)

    def _select_opponents(self):
        """Selects the K best agents from the pool based on ELO."""
        if not self.historical_pool:
            return []
        # Sort the pool by ELO rating in descending order
        sorted_pool = sorted(self.historical_pool, key=lambda x: x['elo'], reverse=True)
        # Return the top K agents
        return sorted_pool[:self.K_OPPONENTS]
    
    def _update_elo(self, winner_elo, loser_elo, k_factor=32):
        """Updates ELO ratings for two players based on a game outcome."""
        expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        new_winner_elo = winner_elo + k_factor * (1 - expected_win)
        new_loser_elo = loser_elo + k_factor * (0 - (1 - expected_win))
        return new_winner_elo, new_loser_elo

    def init_agents(self):
        for x in range(6):
            agent = Agent(player_id = x)
            self.agent_list.append(agent)

    def take_random_legal_actions(self):
        while self.game.check_chance_node() and not self.game.check_terminal():
            self.game.random_chance_node()
    
    def add_agent_topool_from_save(self, path:str):
        temp_agent = Agent(
                player_id=99,  
                game=self.game,
                num_players=self.config["numPlayers"],
                max_action_channels=self.game.max_channels
            )
        print(f"--- Snapshotting agent from iteration {path} into the pool. ---")
        # We use deepcopy to ensure the state_dict is a separate object
        temp_agent.load_policy(file_path=path)
        state_dict = copy.deepcopy(temp_agent.save_policy_dict())
        new_historical_agent = {
                'id': f'loaded_{path}',
                'state_dict': state_dict,
                'elo': 1200, 
                'type': 'historical'
            }
        self.historical_pool.append(new_historical_agent)
        print(f"--- Agent from {path} successfully added. Pool size: {len(self.historical_pool)} ---")


    def to_string(self):
        return self.game.to_string()    
    
    def run_self_play_loop(self,num_iterations=5000, num_episodes_per_iteration=300, resume_from_policy = None):
        """
        Main training loop for self-play with PPO with k self player for heads up.
        """

        if resume_from_policy:
            print(f"--- Resuming training from policy: {resume_from_policy} ---")
            self.load_policy(resume_from_policy)

        self._add_agent_to_pool(0)

        for i in range(num_iterations):
            print(f"--- Iteration {i+1}/{num_iterations} ---")
            # --- 1. Snapshot the agent periodically ---
            if i % 4000 == 0:
                self.save_policy(path=f"poker_ft4_policy_{i}.pth")
            if i % self.SAVE_FREQUENCY == 0:
                self._add_agent_to_pool(i)

            # --- 2. Select opponents for this iteration ---
            opponents = self._select_opponents()
            if not opponents:
                print("Warning: No opponents in the pool. Playing against myself for this iteration.")
                # Fallback to naive self-play if the pool is empty
                opponents = [{'id': 0, 'state_dict': self.poker_agent.save_policy_dict(), 'elo': self.poker_agent_elo}]

            # 1. Generate new self-play trajectories
            for _ in range(num_episodes_per_iteration):
                opponent_data = random.choice(opponents)
                active_opponent = None
                if opponent_data.get('type') == 'nemesis':
                    active_opponent = self.nemesis_agent
                elif opponent_data.get('type') == 'aggressive':
                    active_opponent = self.aggressive_bot
                elif opponent_data.get('type') == 'nit':
                    active_opponent = self.nit_bot
                else:
                    active_opponent = Agent(player_id=1, game=self.game, num_players=self.config["numPlayers"], max_action_channels=self.game.max_channels)
                    active_opponent.load_state_dict(opponent_data['state_dict'])
                    active_opponent.policy_net.eval()

                agents = {0: self.poker_agent, 1: active_opponent}

                # Reset the game for a new episode
                self.game.reset_hand()
                episode_transitions = [] 
                
                while not self.game.check_terminal():

                    # Handle chance nodes (card deals) before a player's turn
                    if self.game.check_chance_node():
                        self.game.random_chance_node()
                        continue
                        
                    # Now it is a player's turn
                    if self.game.check_terminal():
                        break

                    current_player_id = self.game.get_current_player()
                    active_agent = agents[current_player_id]

                    card_tensor = self.game.get_card_tensor(current_player_id)
                    action_tensor = self.game.get_action_tensor()

                    action, log_prob, value = active_agent.select_action(
                        action_tensor= action_tensor,
                        card_tensor= card_tensor,
                        legal_actions=self.game.legal_actions()
                    )
                    
                    #store transition to assign reward to coorect player
                    if current_player_id == 0:
                        episode_transitions.append({
                        "obs": (card_tensor, action_tensor),
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                        "player": current_player_id
                        })

                    # Apply the action
                    self.game.apply_action(action)

                    
                # 2. Learn from the collected data
                final_rewards = self.game.get_rewards()

                # Update elo
                # Only update ELO ratings if the opponent was a historical agent that learns
                if opponent_data.get('type') == 'historical':
                    winner = 0 if final_rewards[0] > final_rewards[1] else 1
                    if winner == 0: # Main agent won
                        new_main_elo, new_opponent_elo = self._update_elo(self.poker_agent_elo, opponent_data['elo'])
                        self.poker_agent_elo = new_main_elo
                        opponent_data['elo'] = new_opponent_elo
                    else: # Opponent won
                        new_opponent_elo, new_main_elo = self._update_elo(opponent_data['elo'], self.poker_agent_elo)
                        self.poker_agent_elo = new_main_elo
                        opponent_data['elo'] = new_opponent_elo
                    
                    # Persist the ELO update back to the main pool
                    for agent_in_pool in self.historical_pool:
                        if agent_in_pool['id'] == opponent_data['id']:
                            agent_in_pool['elo'] = opponent_data['elo']
                            break

                #store transition and rewards
                for transition in episode_transitions:
                    reward = final_rewards[0]
                    # Pass the full transition and its correct reward to the agent
                    self.poker_agent.store_final_transition(transition, reward)

                if episode_transitions: # Avoid adding zero if an episode had no transitions
                    self.poker_agent.add_episode_len(len(episode_transitions))
            
            actor_loss, critic_loss = self.poker_agent.learn()
            
            print(f"Agent trained on {self.poker_agent.prev_mem_len} transitions.")
            print(f"Average Actor Loss: {actor_loss:.4f}, Average Critic Loss: {critic_loss:.4f}")
            print(f"Pool size: {len(self.historical_pool)}")


    def save_policy(self, path:str="poker_policy.pth"):
        self.poker_agent.save_policy(path)

    def load_policy(self, file_path:str="poker_policy.pth"):
        self.poker_agent.load_policy(file_path)

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

                if self.game.check_terminal():
                    break
                
                current_player = self.game.get_current_player()
                card_tensor = self.game.get_card_tensor(current_player)
                action_tensor = self.game.get_action_tensor()
                legal_actions = self.game.legal_actions()

                # Get the action from the agent's policy without exploration
                action, _, _ = poker_agent.select_action(action_tensor, card_tensor, legal_actions)

                # Print the action and game state
                print(f"Player {current_player} takes action: {action}")
                
                # Apply the action to the game
                self.game.apply_action(action)
            
            # Print the final result of the hand
            rewards = self.game.get_rewards()
            print(f"\n--- Episode {episode + 1} Concluded ---")
            print(self.game.to_string())
    
    