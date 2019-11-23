#!/usr/bin/env python3
import numpy as np

import mountain_car_evaluator


def get_decay_rates(
    decay_method, alpha_start, alpha_final, epsilon_start, epsilon_final, n_episodes
):
    if decay_method == "linear":
        alpha_decay_rate = (alpha_start - alpha_final) / n_episodes
        epsilon_decay_rate = (epsilon_start - epsilon_final) / n_episodes
    elif decay_method == "exponential":
        alpha_decay_rate = (alpha_final / alpha_final) ** (1 / (n_episodes - 1))
        epsilon_decay_rate = (epsilon_final / epsilon_start) ** (1 / (n_episodes - 1))
    else:
        alpha_decay_rate = 1
        epsilon_decay_rate = 1

    return alpha_decay_rate, epsilon_decay_rate


def decay(current_rate, decay_rate, decay_method):
    if decay_method == "linear":
        new_rate = current_rate - decay_rate
    elif decay_method == "exponential":
        new_rate = current_rate * decay_rate
    else:
        new_rate = current_rate
    return new_rate


def is_greedy(current_epsilon):
    greedy = np.random.choice([True, False], p=[1 - current_epsilon, current_epsilon])
    return greedy


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument(
        "--render_each", default=None, type=int, help="Render some episodes."
    )

    parser.add_argument("--alpha", default=None, type=float, help="Learning rate.")
    parser.add_argument(
        "--alpha_final", default=None, type=float, help="Final learning rate."
    )
    parser.add_argument(
        "--epsilon", default=None, type=float, help="Exploration factor."
    )
    parser.add_argument(
        "--epsilon_final", default=None, type=float, help="Final exploration factor."
    )
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.weights, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    evaluating = False
    while not evaluating:
        # Perform a training episode
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            greedy = is_greedy(epsilon)
            if greedy:
                action = cd
            # TODO: Choose `action` according to epsilon-greedy strategy

            next_state, reward, done, _ = env.step(action)

            # TODO: Update W values

            state = next_state
            if done:
                break

        # TODO: Decide if we want to start evaluating

        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(
                    np.interp(
                        env.episode + 1,
                        [0, args.episodes],
                        [np.log(args.epsilon), np.log(args.epsilon_final)],
                    )
                )
            if args.alpha_final:
                alpha = (
                    np.exp(
                        np.interp(
                            env.episode + 1,
                            [0, args.episodes],
                            [np.log(args.alpha), np.log(args.alpha_final)],
                        )
                    )
                    / args.tiles
                )

    # Perform the final evaluation episodes
    while True:
        state, done = env.reset(evaluating), False
        while not done:
            # TODO: choose action as a greedy action
            action = ...
            state, reward, done, _ = env.step(action)
