# txhm_pkr

## Introduction
This project is a poker-playing AI designed for fun and to enhance machine learning and reinforcement learning coding skills. It leverages PyTorch for deep learning and OpenSpiel for the poker environment.

## Features
- **Reinforcement Learning**: Implements a PPO-based agent for self-play and policy optimization.
- **PyTorch Integration**: Utilizes PyTorch for building and training neural networks.
- **OpenSpiel Integration**: Uses OpenSpiel to simulate poker games and manage game logic.
- **Rule-Based Opponents**: Includes NemesisBots for efficient training and benchmarking the agent's performance.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd txhm_pkr
   ```
2. Set up a Python virtual environment:
   ```bash
   python3 -m venv pytorch.venv
   source pytorch.venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install torch open_spiel numpy
   ```

## Usage
### Training the Agent
1. Open the `Train.py` script.
2. Adjust parameters as needed (e.g., number of iterations, episodes per iteration).
3. Run the script:
   ```bash
   python Train.py
   ```

### Watching a Playthrough
1. Open the `playthrough_script.py` script.
2. Ensure the desired policy file is loaded.
3. Run the script:
   ```bash
   python playthrough_script.py
   ```

## Libraries Used
- **PyTorch**: For building and training the neural network model.
- **OpenSpiel**: For simulating poker games and managing game logic.

## Future Improvements
- **Benchmarking**: Enhance performance evaluation metrics.
- **6-Player Support**: Extend training and gameplay to support six players.

## License
This project is licensed under the MIT License.

## AlphaHoldEm Inspiration
This project draws significant inspiration from the AlphaHoldEm paper, particularly in the following areas:

- **Training Architecture**: The use of Truncated Clipped Proximal Policy Optimization (PPO) for reinforcement learning is inspired by the methods outlined in AlphaHoldEm.
- **Self-Play Mechanism**: The K-self play strategy, where agents iteratively improve by competing against historical versions of themselves, is modeled after the self-play techniques described in the paper.
- **Network Design**: The pseudo-Siamese network architecture, which processes card and action tensors separately before combining their features, is influenced by the neural network design in AlphaHoldEm.

For more details, refer to the AlphaHoldEm paper: [AlphaHoldEm: Reinforcement Learning for Poker](https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf).

