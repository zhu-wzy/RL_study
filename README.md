# Reinforcement Learning Study & Simulation

This repository contains resources for Reinforcement Learning (RL) study, including theoretical notes and practical implementation demos combining Python and Simulink.

## 📂 Directory Structure Overview

### 1. `demo_python&simulink`
This directory focuses on the implementation of **Model-Free Reinforcement Learning** algorithms applied to a physical system.

* **Environment:**
    * Contains a dynamics model of an **Inverted Pendulum (Cart-Pole)** system constructed using **MATLAB Simscape**.
    ![Simscape Cart-Pole Model](assets/simscape_cartpole.png)
* **Co-Simulation:**
    * Implements a joint simulation framework bridging **Python** (agent/algorithm) and **Simulink** (environment/physics).
* **Algorithms:**
    * The following algorithms have been implemented and tested:
        * **DQN** (Deep Q-Network)
        * **AC** (Actor-Critic)
        * **PPO** (Proximal Policy Optimization)

### 2. `learning_note`
This directory serves as a knowledge base for RL theory.

* Contains comprehensive study notes covering the theoretical foundations of Reinforcement Learning.
* Includes mathematical derivations, core concepts, and summaries of key algorithms.

---
*Created by [ziyan_Wang]*