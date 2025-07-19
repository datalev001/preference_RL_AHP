# Preference Learning and Deep Reinforcement Learning (TD3) for Multi‑Manager Portfolio Strategy Selection
From Human Manager Trajectories to a Unified Adaptive Allocation Policy via AHP‑Guided Preference Modeling and Actor–Critic Optimization

Traditional asset allocation faces challenges in balancing multiple objectives:
Learning from the unique styles and decision patterns of experienced portfolio managers
Managing multiple types of risk, including return, volatility, drawdown, and transaction cost
Adapting allocation decisions in response to changing market conditions

This paper solves these challenges by learning from how different managers adjusted their portfolios in the past. I turn those past decisions into training data for a smart model that can adjust portfolio weights every day, while still following strict risk rules.
I start by extracting key features from each manager's trajectory - returns, volatility, drawdowns, and recovery patterns. The Analytic Hierarchy Process (AHP) sets baseline priorities (e.g., reward returns, penalize drawdowns) using transparent, business-aligned weights. On top of this, I train a preference model (based on pairwise comparisons, like LambdaRank or Bradley–Terry) that captures subtle advantages not reflected in simple ratios - such as smoother recovery after losses.
This combined signal (AHP + learned preference) acts as a custom reward function in a portfolio simulation environment. A TD3 (Twin Delayed DDPG) reinforcement learning agent then learns continuous weight adjustments over trading episodes. Its dual critics and noise control help reduce bias in this sparse reward setup, where feedback only comes from full episodes.
For example, two managers with similar Sharpe ratios may differ in recovery style. The preference model favors the smoother one, and TD3 learns to follow that risk-trimming behavior. The final result is an explainable, adaptive allocation engine that blends human decision logic with deep learning to unlock hidden performance patterns.
