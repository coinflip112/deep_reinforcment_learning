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


arguments = [
    {"dest": "--episodes", "default": 30000, "type": int, "help": "Training episodes."},
    {
        "dest": "--render_each",
        "default": None,
        "type": int,
        "help": "Render some episodes.",
    },
    {"dest": "--alpha", "default": 0.35, "type": float, "help": "Learning rate."},
    {
        "dest": "--alpha_final",
        "default": 0.01,
        "type": float,
        "help": "Final learning rate.",
    },
    {"dest": "--epsilon", "default": 0.5, "type": float, "help": "Exploration factor."},
    {
        "dest": "--epsilon_final",
        "default": 0.00001,
        "type": float,
        "help": "Final exploration factor.",
    },
    {
        "dest": "--decay_method",
        "default": "linear",
        "type": str,
        "help": "Learning rate and Epsilon decay",
    },
    {"dest": "--gamma", "default": 0.99, "type": float, "help": "Discounting factor."},
    {"dest": "--tiles", "default": 8, "type": int, "help": "Number of tiles."},
]

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()

    for argument in arguments:
        param = argument.pop("dest")
        parser.add_argument(param, **argument)

    args = parser.parse_args()

    alpha_decay_rate, epsilon_decay_rate = get_decay_rates(
        decay_method=args.decay_method,
        alpha_start=args.alpha,
        alpha_final=args.alpha_final,
        epsilon_start=args.epsilon,
        epsilon_final=args.epsilon_final,
        n_episodes=args.episodes,
    )
    current_alpha = args.alpha
    current_epsilon = args.epsilon


    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.weights, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    evaluating = False
    while not evaluating:
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            if is_greedy(epsilon):
                action = np.argmax(np.sum(W[state], axis=0))
            else:
                action = np.random.randint(env.states)

            next_state, reward, done, _ = env.step(action)

            # TODO: Update W values
            W[state, action] += alpha * (
                reward
                + args.gamma * np.max(np.sum(W[next_state], axis=0))
                - np.sum(W[next_state, action])
            )

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
    for _ in range(100):
        state, done = env.reset(evaluating), False
        while not done:
            # TODO: choose action as a greedy action
            action = np.argmax(np.sum(W[state], axis=0))
            state, reward, done, _ = env.step(action)
