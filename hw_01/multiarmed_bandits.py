#!/usr/bin/env python3
import argparse
import sys
import operator
import numpy as np
from scipy.special import softmax, expit


class MultiArmedBandits:
    def __init__(self, bandits, episode_length, seed=42):
        self._generator = np.random.RandomState(seed)

        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(self._generator.normal(0.0, 1.0))
        self._done = True
        self._episode_length = episode_length

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError(
                "Cannot step in MultiArmedBandits when there is no running episode"
            )
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = self._generator.normal(self._bandits[action], 1.0)
        return None, reward, self._done, {}


parser = argparse.ArgumentParser()
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episodes", default=100, type=int, help="Training episodes.")
parser.add_argument(
    "--episode_length", default=1000, type=int, help="Number of trials per episode."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument(
    "--mode",
    default="greedy",
    type=str,
    help="Mode to use -- greedy, ucb and gradient.",
)
parser.add_argument(
    "--alpha", default=0, type=float, help="Learning rate to use (if applicable)."
)
parser.add_argument(
    "--c", default=1, type=float, help="Confidence level in ucb (if applicable)."
)
parser.add_argument(
    "--epsilon", default=0.1, type=float, help="Exploration factor (if applicable)."
)
parser.add_argument(
    "--initial",
    default=0,
    type=float,
    help="Initial value function levels (if applicable).",
)


def main(args):
    # Fix random seed
    np.random.seed(args.seed)

    # Create environment
    env = MultiArmedBandits(args.bandits, args.episode_length)
    avg_rewards_per_episode = []
    for episode in range(args.episodes):
        env.reset()
        total_avg_reward = 0
        # TODO: Initialize parameters (depending on mode).
        avg_rewards = {action: args.initial for action in range(args.bandits)}
        counts_taken = {action: args.initial for action in range(args.bandits)}
        done = False
        while not done:
            # TODO: Action selection according to mode
            if args.mode == "greedy":
                action = np.random.choice(
                    ["exploit", "explore"], p=[1 - args.epsilon, args.epsilon]
                )
                if action == "exploit":
                    action = np.argmax(list(avg_rewards.values()))
                else:
                    action = np.random.choice(range(args.bandits))
            elif args.mode == "ucb":
                action = np.argmax(
                    [
                        avg_rewards[action]
                        + args.c * np.sqrt(np.log(env._trials) / counts_taken[action])
                        for action in range(args.bandits)
                    ]
                )
            elif args.mode == "gradient":
                action = np.random.choice(
                    list(avg_rewards.keys()), p=softmax(list(avg_rewards.values()))
                )

            _, reward, done, _ = env.step(action)

            total_avg_reward += 1 / (env._trials) * (reward - total_avg_reward)
            counts_taken[action] += 1

            step_size = 1 / counts_taken[action] if args.alpha == 0.0 else args.alpha

            if args.mode == "gradient":
                softmaxed = softmax(np.array(list(avg_rewards.values())))
                for (action_index, past_action_avg_reward) in avg_rewards.items():
                    if action == action_index:
                        avg_rewards[action] += (
                            step_size
                            * (reward - total_avg_reward)
                            * (1 - softmaxed[action])
                        )
                    else:
                        avg_rewards[action_index] -= (
                            step_size * (reward - total_avg_reward) * softmaxed[action_index]
                        )
            else:
                avg_rewards[action] += step_size * (reward - avg_rewards[action])

        avg_rewards_per_episode.append(total_avg_reward)

    return np.mean(avg_rewards_per_episode), np.std(avg_rewards_per_episode)


if __name__ == "__main__":
    mean, std = main(parser.parse_args())
    # Print the mean and std for ReCodEx to validate
    print("{:.2f} {:.2f}".format(mean, std))
