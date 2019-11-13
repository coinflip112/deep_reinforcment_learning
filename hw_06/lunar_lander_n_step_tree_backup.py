#!/usr/bin/env python3
import numpy as np
import sys
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
    is_greedy = np.random.choice(
        [True, False], p=[1 - current_epsilon, current_epsilon]
    )
    return is_greedy


def get_policy(state, current_epsilon, Q):
    probs = np.ones(env.actions, dtype=float) * current_epsilon / (env.actions-1)
    probs[np.argmax(Q[state][:])] = 1.0 - current_epsilon
    return probs


def get_policy_action(state, current_epsilon, Q):
    probs = get_policy(state, current_epsilon, Q)
    chosen_action = np.random.choice(list(range(env.actions)), p=probs)
    return chosen_action


arguments = [
    {"dest": "--episodes", "default": 20000, "type": int, "help": "Training episodes."},
    {
        "dest": "--render_each",
        "default": None,
        "type": int,
        "help": "Render some episodes.",
    },
    {"dest": "--alpha", "default": 0.3, "type": float, "help": "Learning rate."},
    {
        "dest": "--alpha_final",
        "default": 0.01,
        "type": float,
        "help": "Final learning rate.",
    },
    {"dest": "--epsilon", "default": 0.5, "type": float, "help": "Exploration factor."},
    {
        "dest": "--epsilon_final",
        "default": 0.0001,
        "type": float,
        "help": "Final exploration factor.",
    },
    {
        "dest": "--n_step",
        "default": 20,
        "type": int,
        "help": "Number of steps to tree backup",
    },
    {
        "dest": "--decay_method",
        "default": "exponential",
        "type": str,
        "help": "Learning rate and Epsilon decay",
    },
    {"dest": "--gamma", "default": 1.0, "type": float, "help": "Discounting factor."},
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
    policy = np.array(
        object=[get_policy(state, current_epsilon, Q) for state in range(env.states)]
    )
    # init policy

    for episode in range(args.episodes):
        # Perform a training episode
        state, done = env.reset(), False

        actions = []
        states = []
        rewards = []
        action = get_policy_action(state, current_epsilon, Q)
        actions.append(action)
        states.append(state)
        rewards.append(0)

        T = np.inf
        for t in range(sys.maxsize):
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            if t < T:
                next_state, reward, done, _ = env.step(actions[t])
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1
                else:
                    action = get_policy_action(states[t + 1], current_epsilon, Q)
                    actions.append(action)

            tau = t + 1 - args.n_step
            if tau >= 0:
                if t + 1 >= T:
                    G = rewards[T]
                else:
                    G = rewards[t + 1] + args.gamma * np.sum(
                        policy[states[t + 1], :] * Q[states[t + 1], :]
                    )
                for k in range(min(t, T - 1), tau, -1):
                    G = (
                        rewards[k]
                        + args.gamma
                        * np.sum(
                            [
                                policy[states[k], a] * Q[states[k], a]
                                for a in range(env.actions)
                                if a != states[k]
                            ]
                        )
                        + args.gamma * G * policy[states[k], actions[k]]
                    )

                Q[states[tau], actions[tau]] += current_alpha * (
                    G - Q[states[tau], actions[tau]]
                )
                policy[states[tau], :] = get_policy(states[tau], current_epsilon, Q)
            if tau == T - 1:
                break
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
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False

        while not done:
            action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)
            state = next_state
