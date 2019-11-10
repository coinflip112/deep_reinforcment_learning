#!/usr/bin/env python3
import numpy as np

import lunar_lander_evaluator


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
    is_greedy = np.random.choice([True, False], p=[1 - current_epsilon, current_epsilon])
    return is_greedy

arguments = [
    {"dest": "--episodes", "default": 5500, "type": int, "help": "Training episodes."},
    {
        "dest": "--render_each",
        "default": None,
        "type": int,
        "help": "Render some episodes.",
    },
    {"dest": "--alpha", "default": 0.2, "type": float, "help": "Learning rate."},
    {
        "dest": "--alpha_final",
        "default": 0.1,
        "type": float,
        "help": "Final learning rate.",
    },
    {"dest": "--epsilon", "default": 0.6, "type": float, "help": "Exploration factor."},
    {
        "dest": "--epsilon_final",
        "default": 0.00000000001,
        "type": float,
        "help": "Final exploration factor.",
    },
    {
        "dest": "--n_step",
        "default": 3,
        "type": int,
        "help": "Number of steps to tree backup",
    },
    {
        "dest": "--decay_method",
        "default": "linear",
        "type": str,
        "help": "Learning rate and Epsilon decay",
    },
    {"dest": "--gamma", "default": 1, "type": float, "help": "Discounting factor."},
]


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

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

    # Create the environment
    env = lunar_lander_evaluator.environment()
    current_alpha = args.alpha
    current_epsilon = args.epsilon

    Q = np.zeros(shape=(env.states, env.actions))

    for episode in range(args.episodes):
        current_alpha = decay(
            current_rate=current_alpha,
            decay_rate=alpha_decay_rate,
            decay_method=args.decay_method,
        )
        current_epsilon = decay(
            current_rate=current_epsilon,
            decay_rate=epsilon_decay_rate,
            decay_method=args.decay_method,
        )
        
        # Perform a training episode
        state, done = env.reset(), False
        
        
        actions = []
        states = []
        rewards = []
        actions.append(action)
        states.append(state)
        rewards.append(0)

        T = np.inf
        action = np.random.randint(env.actions)
        t = 0
        while True:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            if t < T:
                next_state, reward, done, _ = env.step(actions[t])

                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1
                else:
                    new_action = np.random.randint(env.actions)
                    actions.append(new_action)

            tau  = t + 1 - args.n_step
            if tau >= 0:
                if t + 1 >= T:
                    G = reward[T]
                else:
                    G = stored_rewards[t + 1] + args.gamma * np.sum(
            policy[stored_states[t + 1]][:] * Q[stored_states[t + 1]][:]
        )           

                    G = rewards[t + 1] + args.gamma * 

    # Perform last 100 evaluation episodes
