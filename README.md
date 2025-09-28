# SimuLife — Agar.io-style RL with DQN + RND (C++/SDL2)

SimuLife is a grid-world/Agar.io-like environment where an organism must forage for food or starve.  
Learning is driven by **Deep Q-Learning (DQN)** with optional **intrinsic motivation via Random Network Distillation (RND)** (Burda et al., 2018).

**Tech stack**
- **Frontend (game & RL loop):** C++17, SDL2
- **Backend (neural_network library):** C++17, Armadillo (linear algebra)
- **Logging & Viz:** CSV logs + Python (matplotlib) scripts

> RND paper: *Exploration by Random Network Distillation*, https://arxiv.org/abs/1810.12894

---

## Quickstart

### 0) Install prerequisites

**macOS**
```bash
brew update
brew install sdl2 sdl2_image sdl2_ttf armadillo cmake
# Python 3 usually comes with venv; otherwise:
brew install python
```

### 1) Build the neural network library
```bash
cd neural_network
make lib
```

This produces a compiled library (.dylib) consumable by the game and moves it to the game directory.

### 2) Build and run the game
```bash
cd ../game
make all
./bin/life
```
**At startup:**
1. Select policy: choose Boltzmann (current supported policy)
2. Toggle RND: enable/disable intrinsic reward
3. Set # episodes: training horizon
4. Start training

### 3) (Optional) Set up Python environment for plots (in game/)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install matplotlib
python3 performance_analysis.py
python3 loss_analysis.py
```