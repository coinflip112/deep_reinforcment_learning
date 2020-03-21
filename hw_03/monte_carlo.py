#!/usr/bin/env python3
import numpy as np

import cart_pole_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", default=5500, type=int, help="Training episodes."
    )
    parser.add_argument(
        "--render_each", default=None, type=int, help="Render some episodes."
    )

    parser.add_argument(
        "--epsilon", default=0.2, type=float, help="Exploration factor."
    )
    parser.add_argument(
        "--epsilon_final", default=0.00001, type=float, help="Final exploration factor."
    )
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    if args.epsilon_final is None:
        decay_method = "no_decay"
        decay_rate = None
    else:
        decay_method = "exponential"
        decay_rate = (args.epsilon_final / args.epsilon) ** (1 / (args.episodes - 1))
    # TODO: Implement Monte-Carlo RL algorithm.
    #
    # The overall structure of the code follows.
    def decay_epsilon(
        start_epsilon,
        current_episode,
        n_episodes,
        end_epsilon=None,
        decay_method="no_decay",
        decay_rate=decay_rate,
    ):
        if decay_method == "linear":
            new_epsilon = start_epsilon - current_episode / n_episodes * (
                start_epsilon - end_epsilon
            )
        if decay_method == "no_decay":
            new_epsilon = start_epsilon
        if decay_method == "exponential":
            new_epsilon = start_epsilon*decay_rate**(current_episode/n_episodes)
        if n_episodes - current_episode <= 200:
            return 0
        return new_epsilon

    Q = np.zeros(shape=(env.states, env.actions))
    C = np.zeros(shape=(env.states, env.actions), dtype=np.int32)
    for episode in range(args.episodes):
        current_epsilon = args.epsilon
        # Perform a training episode
        states = []
        actions = []
        rewards = []
        state, done = env.reset(), False
        current_epsilon = decay_epsilon(
            start_epsilon=args.epsilon,
            current_episode=episode,
            n_episodes=args.episodes,
            end_epsilon=args.epsilon_final,
            decay_method="exponential",
        )
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            is_greedy = np.random.choice(
                [True, False], p=[1 - current_epsilon, current_epsilon]
            )
            if is_greedy:
                action = np.argmax(Q[state, :])
            else:
                action = np.random.randint(low=0, high=env.actions)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            rewards.append(reward)

        G = 0
        for state, action, reward in zip(
            reversed(states), reversed(actions), reversed(rewards)
        ):
            G = args.gamma * G + reward
            C[state, action] += 1
            Q[state, action] += 1 / (C[state, action]) * (G - Q[state, action])

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False

        while not done:
            action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)
            state = next_state
