# 🃏 IIT Pokerbots 2026

A complete setup for the **IIT Pokerbots 2026** competition — includes the bot engine, heuristic test bots, CFR training pipeline, and a build script to package your trained strategy into a submission-ready bot.

---

## 📁 Repository Structure

```
Pokerbot/
├── bot-engine-2026/        # Competition engine & heuristic test bots
│   ├── bot_template.py     # Template for building your final bot
│   ├── heuristic_bots/     # Rule-based bots for local testing
│   └── ...
├── cfr_training/           # Counterfactual Regret Minimization training
│   ├── cfr_train.py        # Main CFR training script
│   ├── build_bot.py        # Bakes strategy + template into final bot
│   └── ...
├── IITPokerbots_PS.pdf     # Official problem statement
└── README.md
```

---

## ⚙️ Setup

**Prerequisites:** Python 3.8+

```bash
git clone https://github.com/Despicoder/Pokerbot.git
cd Pokerbot
pip install -r requirements.txt   # if available
```

---

## 🤖 Heuristic Test Bots

The `bot-engine-2026/` directory contains several rule-based heuristic bots you can use to benchmark your trained bot locally before submission. These bots use hand-strength estimates and fixed betting rules — great for quick sanity checks.

To run a local match between bots, use the engine provided in `bot-engine-2026/`.

---

## 🧠 CFR Training

This repo uses **Counterfactual Regret Minimization (CFR)** to compute an approximate Nash equilibrium strategy for heads-up poker.

### Step 1 — Train the strategy

```bash
python cfr_train.py --iters 300000 --out strategy.pkl
```

| Flag | Description |
|------|-------------|
| `--iters` | Number of CFR iterations (300k is a solid baseline) |
| `--out` | Output path for the serialized strategy file |

Training time scales with `--iters`. More iterations → closer to Nash equilibrium, but takes longer. A `strategy.pkl` file will be saved on completion.

---

## 📦 Building the Submission Bot

Once training is done, bake the strategy into a single self-contained bot file:

### Step 2 — Build the bot

```bash
python build_bot.py --strategy strategy.pkl --template bot_template.py --out bot.py
```

| Flag | Description |
|------|-------------|
| `--strategy` | Path to the trained `.pkl` strategy file |
| `--template` | Bot template that knows how to use the strategy |
| `--out` | Output path for the final submission bot |

The resulting `bot.py` is fully self-contained and ready to submit to the competition engine.

---

## 🚀 Full Pipeline (Quick Reference)

```bash
# 1. Train
python cfr_train.py --iters 300000 --out strategy.pkl

# 2. Build
python build_bot.py --strategy strategy.pkl --template bot_template.py --out bot.py

# 3. Submit bot.py to the competition engine
```

---

## 📄 Problem Statement

See [`IITPokerbots_PS.pdf`](./IITPokerbots_PS.pdf) for the official competition rules and game format.

---

## 🏆 Competition

Built for **IIT Pokerbots 2026**. Post-competition archive — feel free to fork and experiment!

---

*Made by [@Despicoder](https://github.com/Despicoder)*
