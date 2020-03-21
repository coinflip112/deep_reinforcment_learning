#!/usr/bin/env python3
import numpy as np

import mountain_car_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=5500, type=int, help="Training episodes.")
    parser.add_argument(
        "--render_each", default=None, type=int, help="Render some episodes."
    )

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument(
        "--alpha_final", default=0.1, type=float, help="Final learning rate."
    )
    parser.add_argument(
        "--epsilon", default=0.6, type=float, help="Exploration factor."
    )
    parser.add_argument(
        "--epsilon_final",
        default=0.00000000001,
        type=float,
        help="Final exploration factor.",
    )
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()
    if args.epsilon_final is None:
        decay_method = "no_decay"
        decay_rate = None
    else:
        decay_method = "exponential"
        decay_rate = (args.epsilon_final / args.epsilon) ** (1 / (args.episodes - 1))

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
            new_epsilon = start_epsilon * decay_rate ** (current_episode / n_episodes)
        if n_episodes - current_episode <= 300:
            new_epsilon = 0
        return new_epsilon

    def decay_learning_rate(
        start_alpha,
        current_episode,
        n_episodes,
        end_alpha=None,
        decay_method="no_decay",
        decay_rate=decay_rate,
    ):
        if decay_method == "linear":
            new_alpha = start_alpha - current_episode / n_episodes * (
                start_alpha - start_alpha
            )
        if decay_method == "no_decay":
            new_alpha = start_alpha
        if decay_method == "exponential":
            new_alpha = start_alpha * decay_rate ** (current_episode / n_episodes)
        return new_alpha

    Q = np.zeros(shape=(env.states, env.actions))

    # TODO: Implement Q-learning RL algorithm.
    #
    # The overall structure of the code follows.

    for episode in range(args.episodes):
        current_epsilon = args.epsilon
        # Perform a training episode
        state, done = env.reset(), False

        current_epsilon = decay_epsilon(
            start_epsilon=args.epsilon,
            current_episode=episode,
            n_episodes=args.episodes,
            end_epsilon=args.epsilon_final,
            decay_method="linear",
        )
        current_alpha = decay_learning_rate(
            start_alpha=args.alpha,
            current_episode=episode,
            n_episodes=args.episodes,
            end_alpha=args.alpha_final,
            decay_method="linear",
        )

        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            is_greedy = np.random.choice(
                [True, False], p=[1 - current_epsilon, current_epsilon]
            )
            if is_greedy:
                action = np.argmax(Q[state, :])
            else:
                action = np.random.randint(env.actions)
            next_state, reward, done, _ = env.step(action)
            Q[state, action] += current_alpha * (
                reward + args.gamma * np.amax(Q[next_state, :]) - Q[state, action]
            )
            state = next_state

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False

        while not done:
            action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)
            state = next_state
