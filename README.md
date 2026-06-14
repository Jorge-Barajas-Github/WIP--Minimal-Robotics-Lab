# Minimal Robotics Lab

A minimal robotics learning project built around MuJoCo physics simulation and hands-on controller design.

## Purpose

The purpose of this project is to learn robotics and control theory through small, practical simulations.

Instead of only studying equations, this project connects the math to working code. Each controller is implemented, tested, and compared inside a simple MuJoCo environment.

The main focus right now is a pendulum system, starting with basic controllers and gradually moving toward more advanced methods such as reinforcement learning.

## Planned controller progression:

| Order | Controller | Purpose |
|---:|---|---|
| 1 | PD Controller | Learn basic feedback control using position error and velocity damping |
| 2 | LQR Controller | Learn optimal control using state-space models and cost functions |
| 3 | MPC Controller | Learn predictive control by optimizing future actions over a time horizon |
| 4 | Reinforcement Learning Controller | Learn how an agent can improve control behavior through trial and error |

## Pendulum Environment

File:

```text
environments/pendulum_env.py