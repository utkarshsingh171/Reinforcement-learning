# FrozenLake Deep Q-Learning (DQN)

A simple **Deep Q-Learning (DQN)** implementation using **Gymnasium** and **PyTorch** to train an agent to navigate the `FrozenLake-v1` environment.

The project demonstrates the fundamental components of Deep Reinforcement Learning, including:

* Deep Q-Network (DQN)
* Experience Replay
* Epsilon-Greedy exploration
* Target Network
* Bellman equation-based Q-value updates
* Reward tracking
* Model saving and evaluation

## 🚀 Project Overview

The agent learns to navigate a **4×4 FrozenLake environment** and reach the goal while avoiding holes.

The environment is configured with:

```python
gym.make(
    'FrozenLake-v1',
    map_name="4x4",
    is_slippery=False
)
```

With `is_slippery=False`, the environment is deterministic, meaning the agent moves in the intended direction without slipping.

The agent has **16 possible states** and **4 possible actions**:

| Action | Direction |
| ------ | --------- |
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

## 🧠 DQN Architecture

The neural network is intentionally small because FrozenLake has a very small state space.

```text
Input: 16 states
       ↓
Fully Connected Layer
       ↓
16 Hidden Neurons
       ↓
ReLU
       ↓
Output Layer
       ↓
4 Q-values
```

The network predicts the Q-value for each possible action.

For example:

```text
State → [Q(left), Q(down), Q(right), Q(up)]
```

The action with the highest Q-value is selected when the agent exploits its learned policy.

## 🔄 How the Agent Learns

The training process follows this loop:

```text
        ┌──────────────┐
        │ Environment  │
        └──────┬───────┘
               ↓
            State
               ↓
        ┌──────────────┐
        │     DQN      │
        └──────┬───────┘
               ↓
             Action
               ↓
        ┌──────────────┐
        │ Environment  │
        └──────┬───────┘
               ↓
      New State + Reward
               ↓
       Replay Memory
               ↓
        Mini-batch Sample
               ↓
        Optimize DQN
               ↓
       Update Target DQN
```

## 📦 Main Components

### 1. Deep Q-Network

The project defines a simple neural network:

```python
class DQN(nn.Module):
```

It contains:

* Fully connected input layer
* ReLU activation
* Output layer producing Q-values

The network uses the number of environment states as its input size and the number of available actions as its output size.

### 2. Experience Replay

The project stores previous experiences in a replay buffer.

Each experience contains:

```text
(state, action, new_state, reward, terminated)
```

The replay memory uses a fixed maximum size:

```python
replay_memory_size = 1000
```

During training, random mini-batches are sampled from this memory.

### 3. Epsilon-Greedy Exploration

The agent initially explores randomly:

```python
epsilon = 1
```

The agent chooses a random action when:

```text
random < epsilon
```

Otherwise, it selects the action with the highest predicted Q-value.

Epsilon gradually decreases during training.

### 4. Target Network

Two networks are used:

```text
Policy Network
Target Network
```

The target network is periodically synchronized with the policy network.

The synchronization rate is:

```python
network_sync_rate = 10
```

This helps stabilize the Q-learning updates.

## ⚙️ Hyperparameters

The current implementation uses:

| Parameter                |           Value |
| ------------------------ | --------------: |
| Learning Rate            |         `0.001` |
| Discount Factor (γ)      |           `0.9` |
| Target Network Sync Rate |            `10` |
| Replay Memory Size       |          `1000` |
| Mini-batch Size          |            `32` |
| Training Episodes        |          `1000` |
| Environment              | `FrozenLake-v1` |
| Map                      |           `4x4` |
| Slippery                 |         `False` |

These values are defined directly in the implementation.

## 🛠️ Requirements

* Python 3.x
* PyTorch
* Gymnasium
* NumPy
* Matplotlib

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/utkarshsingh171/Reinforcement-learning.git
cd Reinforcement-learning
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install gymnasium numpy matplotlib torch
```

## ▶️ Run the Project

Run the Python file:

```powershell
python reinforcement.py
```

The current configuration trains the agent for:

```python
frozen_lake.train(1000, is_slippery=False)
```

After training, the learned policy is tested for 4 episodes.

## 💾 Output Files

After training, the program generates:

### Trained Model

```text
frozen_lake_dql.pt
```

This contains the trained policy network weights.

### Training Graph

```text
frozen_lake_dql.png
```

The graph contains:

1. Reward progression
2. Epsilon decay

The model and plot are saved automatically after training.

## 📊 Training Visualization

The project tracks rewards collected during each episode and calculates a rolling reward sum over the previous 100 episodes.

It also records the epsilon value throughout training to visualize the exploration-to-exploitation transition.

## 🧪 Testing the Trained Agent

After training, the saved model is loaded and evaluated using:

```python
frozen_lake.test(4, is_slippery=False)
```

During testing, the agent uses the learned policy instead of random exploration.

The environment is rendered so the agent's behavior can be observed.

## 📁 Suggested Project Structure

```text
frozenlake-dqn/
│
├── reinforcement.py
├── frozen_lake_dql.pt
├── frozen_lake_dql.png
└── README.md
```

## 🎯 Learning Objectives

This project was created to understand the practical implementation of **Deep Reinforcement Learning**.

Through this implementation, the following concepts are demonstrated:

* Reinforcement Learning environments
* State and action spaces
* Q-learning
* Deep Q-Networks
* Experience replay
* Exploration vs exploitation
* Target networks
* Neural network optimization
* Model persistence
* Policy evaluation

## 🔮 Future Improvements

Possible extensions include:

* Train on a slippery FrozenLake environment
* Experiment with larger maps
* Tune DQN hyperparameters
* Add reward shaping
* Implement Double DQN
* Implement Dueling DQN
* Add training metrics
* Compare DQN with tabular Q-learning
* Create a custom 2D reinforcement-learning environment
* Apply the same RL framework to a simulated fruit-fly environment

## 📚 Project Context

This project serves as a foundation for understanding how an RL agent can learn through interaction with an environment.

The next step is to move from a simple grid-world environment like FrozenLake toward a **custom 2D environment**, where the agent can learn movement and navigation through rewards rather than using a predefined policy.

## 📄 License

This project is available for educational and research purposes.
