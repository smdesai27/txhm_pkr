import torch
from GameHandler import GameHandler

if __name__ == "__main__":
    # Ensure torch tensors are on CPU if not using a GPU
    # This part can be adjusted based on your system setup
    handler = GameHandler()
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU.")

    handler.add_agent_topool_from_save(path="kself_ft3_policy.pth")
    handler.add_agent_topool_from_save(path="kself_ft2_policy.pth")
    handler.add_agent_topool_from_save(path="kself_policy.pth")

    handler.run_self_play_loop(num_iterations = 50000, num_episodes_per_iteration=300, resume_from_policy= "v2.2_policy.pth")
    handler.save_policy(path="final_policy.pth")