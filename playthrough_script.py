import torch
from GameHandler import GameHandler

if __name__ == "__main__":
    # Ensure torch tensors are on CPU if not using a GPU
    # This part can be adjusted based on your system setup
    handler = GameHandler()
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU.")

    # Uncomment this line to train the agent
    # handler.run_self_play_loop()
    
    # Load the trained policy from the filed
    handler.load_policy(file_path="poker_ft4_policy_8000.pth")
    # Call the new watch_playthrough method to see the agent in action
    handler.watch_playthrough(num_episodes=70)