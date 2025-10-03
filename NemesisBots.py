import random
import numpy as np

class NemesisBot:
    """
    A rule-based agent designed to encourage post-flop play.
    - Pre-flop: Calls with reasonable hands (top 60%), folds weak hands to avoid being too exploitable
    - Post-flop: Acts as a "Check/Raise Bot" to create bluffing opportunities
      and punish mindless continuation betting.
    """
    RAISE_PROBABILITY = 0.3 # 30% chance to raise when facing a bet post-flop
    PREFLOP_CALL_THRESHOLD = 0.40  # Fold bottom 40% of hands pre-flop

    def __init__(self, player_id, max_action_per_round):
        self.player_id = player_id
        self.max_action_per_round = max_action_per_round

    def _estimate_hand_strength(self, card_tensor):
        """
        Simple hand strength estimation based on hole cards.
        Returns a value between 0 and 1, where higher is stronger.
        """
        hole_cards_plane = card_tensor[0]
        ranks = np.where(np.any(hole_cards_plane, axis=0))[0]
        
        if len(ranks) == 0:
            return 0.0
        
        # Pocket pair bonus
        if len(ranks) == 1:
            # Pocket pair - rank goes from 0 (22) to 12 (AA)
            return 0.5 + (ranks[0] / 24.0)  # Range: 0.5 to 1.0
        
        # High card strength
        if len(ranks) == 2:
            high_card = max(ranks)
            low_card = min(ranks)
            # Strong high cards get higher values
            strength = (high_card + low_card) / 24.0  # Range: 0.0 to 1.0
            # Bonus for suited (we don't check suits here, so approximate)
            return strength
        
        return 0.3  # Default moderate strength

    def select_action(self, action_tensor, card_tensor, legal_actions):
        """
        Selects an action based on a "Check/Raise" ruleset by inferring
        the current round from the action_tensor.
        """
        # --- 1. Infer Current Round from Action Tensor ---
        # Find the index of the last channel that has any action recorded
        last_action_indices = np.where(np.any(action_tensor, axis=(1, 2)))[0]

        current_round = 0
        # If any actions have been taken, calculate the round
        if last_action_indices.size > 0:
            last_action_channel = last_action_indices[-1]
            current_round = last_action_channel // self.max_action_per_round
        
        # --- 2. Implement Strategy based on the Round ---
        action = 0 # Default to fold
        CAN_CHECK_OR_CALL = 1 in legal_actions

        if current_round == 0:
            # --- FIXED PRE-FLOP LOGIC: Fold weak hands, call with decent hands ---
            hand_strength = self._estimate_hand_strength(card_tensor)
            
            if hand_strength < self.PREFLOP_CALL_THRESHOLD:
                # Weak hand - fold if facing a bet
                if 0 in legal_actions:  # Facing a bet
                    action = 0  # Fold
                elif CAN_CHECK_OR_CALL:
                    action = 1  # Check for free
            else:
                # Decent hand - call to see flop
                if CAN_CHECK_OR_CALL:
                    action = 1
                else:
                    action = 0  # Fold if can't call
        else:
            # --- POST-FLOP LOGIC: Be a Check/Raise Bot ---
            # Heuristic: If fold (0) is NOT a legal action, it means we can check for free.
            is_first_to_act = 0 not in legal_actions

            if is_first_to_act:
                # If we are first to act, always check to invite a bet.
                action = 1
            else:
                # We are facing a bet. Time to be unpredictable.
                raise_actions = [a for a in legal_actions if a > 1]
                
                # Decide whether to raise or just call
                if raise_actions and random.random() < self.RAISE_PROBABILITY:
                    # 30% of the time, make a random-sized raise.
                    action = random.choice(raise_actions)
                elif CAN_CHECK_OR_CALL:
                    # Otherwise, just call.
                    action = 1
                else:
                    # If we can't call or raise, we must fold.
                    action = 0

        # --- 3. Final safety check ---
        # Ensure the chosen action is legal, otherwise pick a random valid one.
        if action not in legal_actions:
            non_fold_actions = [a for a in legal_actions if a > 0]
            if non_fold_actions:
                action = random.choice(non_fold_actions)
            else:
                action = 0 # Can only fold

        # Return dummy values for log_prob and value as this agent does not learn
        return action, 0.0, 0.0


class AggressiveBlufferBot:
    """
    A rule-based agent that plays a hyper-aggressive style.
    Its goal is to put maximum pressure on the opponent at all times.

    - Pre-flop: Raises or re-raises with a high frequency to steal pots early.
    - Post-flop: Bets (continuation-bets) or raises frequently to represent strength,
      regardless of its actual hand value.
    """
    PREFLOP_RAISE_PROBABILITY = 0.70  # 70% chance to open-raise instead of limping
    PREFLOP_RERAISE_PROBABILITY = 0.40 # 40% chance to 3-bet when facing a raise
    POSTFLOP_BET_PROBABILITY = 0.80    # 80% chance to bet if checked to
    POSTFLOP_RAISE_PROBABILITY = 0.50  # 50% chance to raise if bet into

    def __init__(self, player_id, max_action_per_round):
        self.player_id = player_id
        self.max_action_per_round = max_action_per_round

    def select_action(self, action_tensor, card_tensor, legal_actions):
        """
        Selects an action based on a hyper-aggressive ruleset.
        """
        # 1. Infer Current Round from Action Tensor 
        last_action_indices = np.where(np.any(action_tensor, axis=(1, 2)))[0]
        current_round = 0
        if last_action_indices.size > 0:
            last_action_channel = last_action_indices[-1]
            current_round = last_action_channel // self.max_action_per_round
        
        # 2. Implement Strategy 
        action = 0  # Default to fold
        CAN_CALL = 1 in legal_actions
        raise_actions = [a for a in legal_actions if a > 1]
        can_raise = bool(raise_actions)

        if current_round == 0:
            # pRE-FLOP LOGIC: Be aggressive 
            is_facing_bet = 0 in legal_actions # If fold is an option, someone has bet/raised

            if not is_facing_bet: # First to act or facing limpers
                if can_raise and random.random() < self.PREFLOP_RAISE_PROBABILITY:
                    action = random.choice(raise_actions)
                elif CAN_CALL:
                    action = 1
            else: # Facing a raise
                if can_raise and random.random() < self.PREFLOP_RERAISE_PROBABILITY:
                    action = random.choice(raise_actions)
                elif CAN_CALL:
                    action = 1
        else:
            # POST-FLOP LOGIC: Constant pressure 
            is_first_to_act = 0 not in legal_actions

            if is_first_to_act:
                # If we are first to act, bet with high probability
                if can_raise and random.random() < self.POSTFLOP_BET_PROBABILITY:
                    action = random.choice(raise_actions)
                else:
                    action = 1 # Check
            else:
                # Facing a bet, raise 50% of the time
                if can_raise and random.random() < self.POSTFLOP_RAISE_PROBABILITY:
                    action = random.choice(raise_actions)
                elif CAN_CALL:
                    action = 1

        # saftey chekc
        if action not in legal_actions:
            # Fallback if preferred action is illegal (e.g. can't cover a raise)
            if CAN_CALL:
                action = 1
            else:
                action = 0 # Can only fold

        # Return dummy values for log_prob and value as this agent does not learn
        return action, 0.0, 0.0
    

class NitBot:
    """
    A rule-based agent that plays an extremely tight ("nitty") style.
    It only voluntarily puts money in the pot with premium hands.

    - Pre-flop: Only plays high pocket pairs (Tens or better) and high-card combos (AQ+).
      Folds everything else unless it can check for free.
    - Post-flop: Plays aggressively if it has a strong hand (top pair or better),
      otherwise, it plays "fit-or-fold" and gives up easily.
    """
    def __init__(self, player_id, max_action_per_round):
        self.player_id = player_id
        self.max_action_per_round = max_action_per_round
        self.played_premium_preflop = False

    def _is_premium_preflop(self, card_tensor):
        """ Checks the card_tensor for a premium starting hand. """
        # Get the ranks of the hole cards (0-12 for 2-A)
        hole_cards_plane = card_tensor[0]
        ranks = np.where(np.any(hole_cards_plane, axis=0))[0]

        # Case 1: Pocket Pair (e.g., QQ, KK, AA)
        # Rank 8 is Ten, so we check for Tens or better.
        if len(ranks) == 1 and ranks[0] >= 8: 
            return True

        # Case 2: Two High Cards (e.g., AK, AQ)
        # Rank 10 is Queen, Rank 11 is King, Rank 12 is Ace.
        # Checks for AQ, AK, KQ (as KQ is also very strong heads-up)
        if len(ranks) == 2 and np.all(ranks >= 10):
            return True
            
        return False

    def _has_strong_postflop_hand(self, card_tensor):
        """ Checks for top pair or better post-flop. """
        hole_cards_plane = card_tensor[0]
        board_cards_plane = card_tensor[4] # Plane with all public cards
        
        hole_ranks = np.where(np.any(hole_cards_plane, axis=0))[0]
        board_ranks = np.where(np.any(board_cards_plane, axis=0))[0]

        # Simple check: did one of our hole cards pair the board?
        for rank in hole_ranks:
            if rank in board_ranks:
                return True # We have at least a pair.

        return False


    def select_action(self, action_tensor, card_tensor, legal_actions):
        """
        Selects an action based on a tight, nitty ruleset.
        """
        # 1. Infer Current Round and Reset Memory 
        last_action_indices = np.where(np.any(action_tensor, axis=(1, 2)))[0]
        current_round = 0
        if last_action_indices.size > 0:
            last_action_channel = last_action_indices[-1]
            current_round = last_action_channel // self.max_action_per_round
        else:
             # This is the first action of the hand, reset memory
            self.played_premium_preflop = False

        # 2. Implement Strategy ---
        action = 0 # Default to fold
        CAN_CHECK_OR_CALL = 1 in legal_actions
        raise_actions = [a for a in legal_actions if a > 1]

        if current_round == 0:
            # PRE-FLOP LOGIC: Premium hands only ---
            if self._is_premium_preflop(card_tensor):
                self.played_premium_preflop = True
                # If we have a premium hand, always raise if possible.
                if raise_actions:
                    action = random.choice(raise_actions)
                else:
                    action = 1 # Just call if no raise option
            else:
                # Not a premium hand. Fold to any bet, otherwise check.
                if CAN_CHECK_OR_CALL and 0 not in legal_actions:
                    action = 1 # Can check for free
                else:
                    action = 0 # Fold
        else:
            # POST-FLOP LOGIC: Fit or Fold ---
            # If we started with a premium pair (like AA), we are still confident.
            # Or if we hit a good pair on the board.
            if self.played_premium_preflop or self._has_strong_postflop_hand(card_tensor):
                # Strong hand: Bet or raise.
                 if raise_actions:
                    action = random.choice(raise_actions)
                 elif CAN_CHECK_OR_CALL:
                    action = 1
            else:
                # We missed. Check if we can, otherwise fold.
                if CAN_CHECK_OR_CALL and 0 not in legal_actions:
                    action = 1 # Can check for free
                else:
                    action = 0 # Fold to any bet

        # 3. Final safety check ---
        if action not in legal_actions:
            action = 0 if 0 in legal_actions else random.choice(legal_actions)

        return action, 0.0, 0.0


class BalancedTAGBot:
    """
    A Tight-Aggressive (TAG) bot that plays a balanced, GTO-approximation strategy.
    
    - Pre-flop: Plays premium hands aggressively, good hands cautiously, folds weak hands
    - Post-flop: Uses hand strength and pot odds to make balanced decisions
    - Bet sizing: Uses varied bet sizes based on situation
    """
    PREMIUM_THRESHOLD = 0.75  # Top 25% of hands
    PLAYABLE_THRESHOLD = 0.50  # Top 50% of hands
    POSTFLOP_STRENGTH_THRESHOLD = 0.60
    BLUFF_FREQUENCY = 0.15  # 15% bluff frequency
    
    def __init__(self, player_id, max_action_per_round):
        self.player_id = player_id
        self.max_action_per_round = max_action_per_round
        self.preflop_strength = 0.0
    
    def _estimate_hand_strength(self, card_tensor):
        """Estimate hand strength from 0 to 1."""
        hole_cards_plane = card_tensor[0]
        ranks = np.where(np.any(hole_cards_plane, axis=0))[0]
        
        if len(ranks) == 0:
            return 0.0
        
        # Pocket pair
        if len(ranks) == 1:
            # AA=12, 22=0, scale from 0.6 to 1.0
            return 0.6 + (ranks[0] / 30.0)
        
        # Two cards
        if len(ranks) == 2:
            high_card = max(ranks)
            low_card = min(ranks)
            gap = high_card - low_card
            
            # High cards are strong
            base_strength = (high_card + low_card) / 24.0
            
            # Penalize large gaps
            gap_penalty = gap * 0.02
            
            return max(0.0, min(1.0, base_strength - gap_penalty))
        
        return 0.3
    
    def _has_made_hand(self, card_tensor):
        """Check if we have a pair or better."""
        hole_cards_plane = card_tensor[0]
        board_cards_plane = card_tensor[4]
        
        hole_ranks = np.where(np.any(hole_cards_plane, axis=0))[0]
        board_ranks = np.where(np.any(board_cards_plane, axis=0))[0]
        
        # Check for pair
        for rank in hole_ranks:
            if rank in board_ranks:
                return True
        
        # Check if we have a pocket pair
        if len(hole_ranks) == 1:
            return True
        
        return False
    
    def select_action(self, action_tensor, card_tensor, legal_actions):
        """Balanced TAG strategy."""
        # Infer current round
        last_action_indices = np.where(np.any(action_tensor, axis=(1, 2)))[0]
        current_round = 0
        if last_action_indices.size > 0:
            last_action_channel = last_action_indices[-1]
            current_round = last_action_channel // self.max_action_per_round
        else:
            self.preflop_strength = self._estimate_hand_strength(card_tensor)
        
        action = 0
        CAN_CHECK_OR_CALL = 1 in legal_actions
        raise_actions = [a for a in legal_actions if a > 1]
        is_facing_bet = 0 in legal_actions
        
        if current_round == 0:
            # PRE-FLOP: Tight-aggressive approach
            hand_strength = self._estimate_hand_strength(card_tensor)
            
            if hand_strength >= self.PREMIUM_THRESHOLD:
                # Premium hands: raise aggressively
                if raise_actions:
                    # Prefer smaller raises pre-flop
                    action = raise_actions[0] if len(raise_actions) > 0 else raise_actions[0]
                elif CAN_CHECK_OR_CALL:
                    action = 1
            elif hand_strength >= self.PLAYABLE_THRESHOLD:
                # Playable hands: call or small raise
                if is_facing_bet:
                    if CAN_CHECK_OR_CALL:
                        action = 1  # Call
                else:
                    # First to act: mix of raise and check
                    if raise_actions and random.random() < 0.4:
                        action = raise_actions[0]
                    elif CAN_CHECK_OR_CALL:
                        action = 1
            else:
                # Weak hands: fold to bets, check if free
                if is_facing_bet:
                    action = 0  # Fold
                elif CAN_CHECK_OR_CALL:
                    action = 1  # Check
        
        else:
            # POST-FLOP: Balanced play
            has_hand = self._has_made_hand(card_tensor)
            
            if has_hand or self.preflop_strength >= self.PREMIUM_THRESHOLD:
                # Strong hand: bet for value
                if not is_facing_bet:
                    # First to act: bet frequently
                    if raise_actions and random.random() < 0.7:
                        # Use varied bet sizes
                        action = random.choice(raise_actions)
                    elif CAN_CHECK_OR_CALL:
                        action = 1
                else:
                    # Facing a bet: call or raise
                    if raise_actions and random.random() < 0.3:
                        action = random.choice(raise_actions[:2])  # Don't all-in often
                    elif CAN_CHECK_OR_CALL:
                        action = 1
            else:
                # Weak hand: bluff occasionally or fold
                if not is_facing_bet:
                    # First to act: bluff sometimes
                    if raise_actions and random.random() < self.BLUFF_FREQUENCY:
                        action = raise_actions[0]  # Small bluff
                    elif CAN_CHECK_OR_CALL:
                        action = 1  # Check
                else:
                    # Facing a bet: mostly fold, occasionally bluff-raise
                    if raise_actions and random.random() < 0.05:
                        action = raise_actions[0]
                    else:
                        action = 0  # Fold
        
        # Safety check
        if action not in legal_actions:
            if CAN_CHECK_OR_CALL:
                action = 1
            else:
                action = 0
        
        return action, 0.0, 0.0
