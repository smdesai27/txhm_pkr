import pyspiel, random  
import numpy as np 

class GameWrapper:

    round_len_heuristic = 10
    suit_dict = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    rank_dict = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

    def __init__(self, config: dict):
        self.game = pyspiel.load_game("universal_poker", config)
        self.state = self.game.new_initial_state()

        self.num_players = self.game.num_players()
        self.max_action_per_round = self.num_players * GameWrapper.round_len_heuristic
        self.max_channels = 4 * self.max_action_per_round
        
        self.reset_alpha_holdem_action_tensor()
        self.actions_per_round_ct = 0


    # while state.is_chance_node():
    # Run ^^ Randomally auto-resolve all cahnce nodes (deal, blind, etc)
    def random_chance_node(self):
        outcomes = self.state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = random.choices(actions, probs)[0]
        self.apply_chance_action(action)

    # return true if is chance node
    def check_chance_node(self):
        return self.state.is_chance_node()

    # returns list of legal action ints
    def legal_actions(self):
        return self.state.legal_actions()
    
    def apply_chance_action(self, action_int: int):
        self.state.apply_action(action_int)
        self.actions_per_round_ct = 0
    
    def get_round_num(self):
        return int(self.state.information_state_string(1).split("[Round ")[1].split("]")[0])

    # takes action int and applys it to currnet state
    def apply_action(self, action_int : int, player_id: int = None):
        
        if player_id is None:
            player_id = self.state.current_player()

        # print(f'action int is {action_int}')
        # print(f"action string to bet {self.state.action_to_string(int(player_id), int(action_int))}")

        round_number = self.get_round_num()
        l_a = np.array(self.legal_actions())
        self.update_alpha_holdem_action_tensor(player_id=player_id, action_int=action_int, round_number=round_number, legal_actions=l_a)

        self.state.apply_action(action_int)
        self.actions_per_round_ct += 1

    # returns true is game is over
    def check_terminal(self):
        return self.state.is_terminal()
    
    def get_current_player(self):
        return self.state.current_player()
    
    def to_string(self):
        return self.state

    def get_card_tensor(self, player_id: int):
        # indexes: hole = 0, flop = 1, turn = 2, river = 3, all_public = 4, all_pub_and_priv = 5
        card_tensor = np.zeros((6,4,13), dtype=np.int8)

        # do hole cards
        priv_seq_str = self.state.information_state_string(player_id).split("[Private: ")[1].split("]")[0]
        for c in range(0,len(priv_seq_str),2):
            two_card_chars = priv_seq_str[c:c+2]
            col_idx = GameWrapper.rank_dict[two_card_chars[0]]
            row_idx = GameWrapper.suit_dict[two_card_chars[1]]
            card_tensor[0,row_idx,col_idx] = 1


        # do public cards
        action_seq_str = self.state.information_state_string(player_id).split("[Public: ")[1].split("]")[0]
        count = 1
        for c in range(0,len(action_seq_str),2):
            two_card_chars = action_seq_str[c:c+2]
            col_idx = GameWrapper.rank_dict[two_card_chars[0]]
            row_idx = GameWrapper.suit_dict[two_card_chars[1]]
            
            if count <= 3:
                card_tensor[1,row_idx,col_idx] = 1

            elif count == 4:
                card_tensor[2,row_idx,col_idx] = 1
            
            elif count == 5:
                card_tensor[3,row_idx,col_idx] = 1
            
            count+=1
        
        card_tensor[4] = np.logical_or.reduce(card_tensor[1:4]).astype(np.int8)
        card_tensor[5] = np.logical_or.reduce(card_tensor[0:5]).astype(np.int8)

        return card_tensor.copy()
    
    def test_method(self):
        print(self.state)
        s = self.state.information_state_string(3).split("[Public: ")[1].split("]")[0]

    
    def decode_action_from_two_bits(self, full_array: np.array):
        """
        Decodes a sequence of two-bit action encodings into integers.
        full array of 2 set bits
        bit1_array: A NumPy array containing the first bit of each action (0.0 or 1.0).
        bit2_array: A NumPy array containing the second bit of each action (0.0 or 1.0).
        Returns:
            A NumPy array of the decoded actions (0-3).
        """
        # Even-indexed elements (0, 2, 4, ...)
        bit1_array = full_array[::2]
        # Odd-indexed elements (1, 3, 5, ...)
        bit2_array = full_array[1::2]

        decoded_array = (bit1_array * 2) + bit2_array
        return decoded_array

    
    def reset_alpha_holdem_action_tensor(self):
        self.action_tensor = np.zeros((self.max_channels, self.num_players + 2 , 5), dtype = np.int8)


    def update_alpha_holdem_action_tensor(self, action_int: int, player_id: int, round_number: int, legal_actions : np.array):  

        if self.actions_per_round_ct >= self.max_action_per_round:
            print(f"--- TENSOR OVERFLOW WARNING ---")
            print(f"Exceeded max_action_per_round limit of {self.max_action_per_round} with heuristic {GameWrapper.round_len_heuristic}.")
            print(f"Action {action_int} by player {player_id} in round {round_number} was not recorded in the action tensor.")
            print(f"----------------------")
            return # Exit the function to prevent the crash
        # --- END OF SAFETY CHECK ---

        # TODO IS PLAYER ID GONNA START AT 0 OR 1
        sum_row_idx = self.num_players
        legal_row_idx = self.num_players + 1
 
        self.action_tensor[round_number * self.max_action_per_round + self.actions_per_round_ct, player_id, action_int] = 1

        self.action_tensor[round_number * self.max_action_per_round + self.actions_per_round_ct, sum_row_idx, :] = \
            np.logical_or.reduce(self.action_tensor[round_number * self.max_action_per_round + self.actions_per_round_ct, 0: sum_row_idx, :]).astype(np.int8)

        self.action_tensor[round_number * self.max_action_per_round + self.actions_per_round_ct, legal_row_idx, :][legal_actions] = 1

    def get_action_tensor(self):
        return self.action_tensor.copy()
    
    def get_rewards(self):
        return self.state.returns()
    
    def reset_hand(self):
        self.state = self.game.new_initial_state()
        self.reset_alpha_holdem_action_tensor()
        self.actions_per_round_ct = 0