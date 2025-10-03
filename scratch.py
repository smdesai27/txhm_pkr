import pyspiel, random
from GameWrapper import GameWrapper

config = {
    "betting": "nolimit",
    "numPlayers": 2,
    "numRounds": 4,
    "blind": "100 50",
    "firstPlayer": "2 1 1 1",
    "numSuits": 4,
    "numRanks": 13,
    "numBoardCards": "0 3 1 1",
    'numHoleCards' : 2,
    "stack": "20000 20000",
    "bettingAbstraction": "fchpa",
    "potSize": 0,
    "boardCards": "",
    "maxRaises": "255"

}

game = GameWrapper(config)

# Auto-resolve all chance nodes (deals, blinds, etc.)
while game.check_chance_node():
    game.random_chance_node()

# Play a random hand to the end, printing state at each step
while not game.check_terminal():
    if game.check_chance_node():
        game.random_chance_node()

    
        
    else:
        print(game.to_string())
        legal_actions = game.legal_actions()
        print(f"legal actions are {legal_actions}")


        if(game.get_current_player() == 1):
            action = int(input("Choose an action index :: "))
        
        else:
            action = random.choice(legal_actions)
        
        game.apply_action(action)
        print(f"round num is + {game.get_round_num()}")
    
    
    
print(game.to_string())