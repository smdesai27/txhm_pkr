import torch
from GameHandler import GameHandler

if __name__ == "__main__":
    # Ensure torch tensors are on CPU if not using a GPU
    # This part can be adjusted based on your system setup
    handler = GameHandler()
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU.")

    handler.run_self_play_loop(num_iterations = 15000, num_episodes_per_iteration=300)
    handler.save_policy()