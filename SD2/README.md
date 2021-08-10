# SD2
Code for Sub-dynamics Discovery (SD2).

## Installation
Based on the environment installed in ED2-Dreamer and ED2-MBPO.

## Usage
First analysis action-state relation by running:
For dmc suite tasks:
```
python dmc_control.py
```
For dmc suite tasks with visual input:
```
python dmc_control_visual.py
```
For gym-mujoco tasks:
```
python gym_mujoco.py
```

Then clustering the action dimensions with input the action-state relations:
```
python clustering.py --path ./rela_gym/Walker2d_v2/0.xlsx
```
We will output rela(G_i, G_j) each clustering step, and we can the result when rela(G_i, G_j) < eta
