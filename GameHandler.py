from GameWrapper import GameWrapper
from Agent import Agent
import random
import torch # type: ignore

class GameHandler:

    num_players = 2

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
        self.poker_agent = Agent(player_id=0, game = self.game, num_players=GameHandler.config["numPlayers"])

    def init_agents(self):
        for x in range(6):
            agent = Agent(player_id = x)
            self.agent_list.append(agent)

    def take_random_legal_actions(self):
        while self.game.check_chance_node() and not self.game.check_terminal():
            self.game.random_chance_node()  

    def to_string(self):
        return self.game.to_string()    
    
    def run_self_play_loop(self,num_iterations=5000, num_episodes_per_iteration=100):
        """
        Main training loop for self-play with PPO.
        """

        for i in range(num_iterations):
            print(f"--- Iteration {i+1}/{num_iterations} ---")
            
            # 1. Generate new self-play trajectories
            for _ in range(num_episodes_per_iteration):
                # Reset the game for a new episode
                self.game.reset_hand()
                episode_transitions = [] ## TODO IMPLEMENT FIX FOR OPPOSING PLAYER GETTING REWARDS FOR ONE 
                
                while not self.game.check_terminal():
                    # Handle chance nodes (card deals) before a player's turn
                    while self.game.check_chance_node():
                        self.game.random_chance_node()
                        
                    # Now it is a player's turn
                    if self.game.check_terminal():
                        break

                    current_player = self.game.get_current_player()
                    card_tensor = self.game.get_card_tensor(current_player)
                    action_tensor = self.game.get_action_tensor()
                    legal_actions = self.game.legal_actions()

                    # Select an action
                    action, log_prob, value = self.poker_agent.select_action(action_tensor, card_tensor, legal_actions)
                    # Store the transition
                    self.poker_agent.reward_free_store_transition((card_tensor, action_tensor), action, log_prob, value)
                    # Apply the action
                    self.game.apply_action(action)
                    
                # 2. Learn from the collected data
                reward = self.game.get_rewards()[current_player]
                self.poker_agent.store_transition_with_reward(reward)
            
            actor_loss, critic_loss = self.poker_agent.learn()
            
            print(f"Agent trained on {self.poker_agent.prev_mem_len} transitions.")
            print(f"Average Actor Loss: {actor_loss:.4f}, Average Critic Loss: {critic_loss:.4f}")


    def save_policy(self, path:str="poker_policy.pth"):
        self.poker_agent.save_policy(path)

    def load_policy(self, path:str="poker_policy.pth"):
        self.poker_agent.load_policy(path)

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
    
    