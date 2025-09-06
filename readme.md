

# RL-doom-in-30x30

A minimalist reinforcement learning experiment where a CNN-powered agent battles an ASCII-art boss in a 30×30 grid world. Think of it as *Doom demake meets reinforcement learning*.

## 🎮 Game Environment

- **Grid size:** 30×30
- **Player:** `@` (controlled by the agent)
- **Boss:** `#` (stationary at the top of the grid)
- **Player laser:** `!`
- **Boss laser:** `|` (inactive in this version)
- **Actions:**
  - `0` → Shoot
  - `1` → Move Left
  - `2` → Move Right

The game runs for a maximum of 200 turns per episode or until the boss is defeated.

## 🧠 The Agent (EYES)

The agent is a tiny CNN designed to process the ASCII grid:

```
Input:  (1, 30, 30)
Conv2D → MaxPool → ReLU
Conv2D → MaxPool → ReLU
Flatten → Linear(3)
```

It outputs logits for the 3 possible actions, sampled using a Categorical distribution. Training uses a REINFORCE-style policy gradient.

## 🏆 Reward System

- +1 per step (base reward)
- **Win (boss defeated):** `+50` bonus after flipping accumulated reward negative
- **Timeout (200 turns):** accumulated reward flipped negative and penalized `-30`

This makes survival alone not enough — the agent must learn to shoot accurately.

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-username/RL-doom-in-30x30.git
cd RL-doom-in-30x30
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training:
```bash
python train.py
```

Optionally enable rendering by setting `RENDER_GAME = True` in `train.py`.

## 📂 Project Structure

```
RL-doom-in-30x30/
├── train.py            # Main training loop
├── environment.py      # Game environment implementation
├── agent.py            # CNN agent implementation
├── models/              # Saved models (.pth)
├── logs/               # Training logs
├── feature_maps/        # CNN visualizations
└── README.md
```

## 📊 Example Output

```
--- Episode 100 ---
Total reward: 42.5
Steps in episode: 125
Result: win
Loss: 0.4321
```

## 🔮 Future Enhancements

- Enable boss attacks with active lasers
- Add difficulty levels (boss movement, multi-boss fights)
- Implement advanced RL algorithms (DQN, PPO, A2C)
- Record gameplay as GIFs

## 📈 Visualizing Features

To visualize CNN feature maps:

```bash
python visualize_features.py
```

This will generate feature maps for both convolutional layers in the `feature_maps/` directory.

## ⚡ Inspiration

Inspired by ASCII roguelikes, tiny RL projects, and the idea of running Doom on everything — why not in a 30×30 grid?

---

Created with ❤️ for reinforcement learning enthusiasts and minimalists alike.