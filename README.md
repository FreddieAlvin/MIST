# Project MIST: Multimodal Intelligent Sensor-fusion for Trajectory-navigation

**Course:** Introduction to Intelligent Robotics  
**Objective:** Investigation of RL agent robustness in low-visibility navigation environments.

## Overview
Project MIST (Multimodal Intelligent Sensor-fusion for Trajectory-navigation) explores how autonomous agents can maintain navigation reliability in degraded sensor conditions. Using a Pioneer 3AT robot in the Webots simulator, we test agents against simulated environmental challenges such as fog-induced LiDAR backscattering.

The core of our research is a "speed-to-visibility" policy—an RL-driven mechanism that dynamically balances navigation safety and operational efficiency by fusing noisy distance data with kinematic cues.

## Core Features
* **Sensor Fusion Architecture:** Implements a `MultiInputPolicy` using Stable-Baselines3.
    * **Kinematic/Proximity Stream:** 16-sector LiDAR array and kinematic sensor vectors processed via `MLP` to handle velocity, heading, and proximity data.
* **Robustness Training:** Incorporates a noise curriculum that introduces increasing levels of sensor interference (simulating fog) during the training phase.
* **Navigation Modes:** Supports three distinct evaluation modes:
    * **PURE_DSTAR:** D* Lite incremental path planning for baseline comparison.
    * **PURE_APF:** Artificial Potential Fields for reactive obstacle avoidance.
    * **PPO:** Trained reinforcement learning agent for robust, adaptive navigation.

## Technical Stack
* **Simulator:** Webots
* **Framework:** Stable-Baselines3
* **Environment:** Gymnasium
* **Language:** Python 3.11

## Quick Start Guide

### 1. Training the PPO Agent
Train the policy using the sensor-fusion architecture:
```bash
python3 src/agents/train_ppo.py
```

### 2. Evaluating Navigation Modes
Run the controller in your preferred navigation mode (e.g., PPO):
```bash
MIST_NOISE=0.00 MIST_NAV_MODE=PPO python3 src/controllers/mist_controller.py
```

### 3. Analyzing Performance
Generate comparison charts (time per lane, distance travelled, completion rate):
```bash
python3 src/utils/plot_results.py
```

## Research Objectives
Our goal is to quantify the "confidence threshold" at which an agent should switch from high-speed navigation to cautious, sensor-dependent movement. By fusing multimodal data, the MIST agent learns to weigh proximity data against motion state, allowing it to perform effectively even when sensory inputs are severely corrupted by environmental noise.
Created for the Introduction to Intelligent Robotics course.

