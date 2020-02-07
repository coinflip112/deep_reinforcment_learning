# 00c25fc3-eb3a-11e8-b0fd-00505601122b
# dcfac0e3-1ade-11e8-9de3-00505601122b
# 7e271e04-8848-11e7-a75c-005056020108

import collections

import numpy as np
import tensorflow as tf
from tensorflow import keras

import random

import cart_pole_evaluator


class Network:
    def __init__(self, env, args, compile_network=True):
        # Create a suitable network

        self.model = keras.Sequential(
            layers=[
                keras.layers.Dense(args.hidden_layer_size, activation="relu"),
                keras.layers.Dense(env.actions),
            ]
        )
        if compile_network:
            self.model.compile(
                optimizer="adam", loss="mse", experimental_run_tf_function=False
            )

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, including the index of the action to which
    #   the new q_value belongs
    def train(self, states, targets):
        self.model.train_on_batch(states, targets)

    def predict(self, states):
        return self.model(states)

    def load(self, weights_file):
        self.model.load_weights(weights_file)

    def dump(self, weights_file):
        self.model.save_weights(weights_file)


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=42, type=int, help="Batch size.")
    parser.add_argument(
        "--episodes", default=700, type=int, help="Episodes for epsilon decay."
    )
    parser.add_argument(
        "--epsilon", default=0.4, type=float, help="Exploration factor."
    )
    parser.add_argument(
        "--epsilon_final", default=0.01, type=float, help="Final exploration factor."
    )
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument(
        "--hidden_layers", default=1, type=int, help="Number of hidden layers."
    )
    parser.add_argument(
        "--hidden_layer_size", default=20, type=int, help="Size of hidden layer."
    )
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--render_each", default=0, type=int, help="Render some episodes."
    )
    parser.add_argument(
        "--threads", default=1, type=int, help="Maximum number of threads to use."
    )
    parser.add_argument("--training", default=False, type=bool, help="Is this trainig?")
    parser.add_argument(
        "--save", default=False, type=bool, help="Should trained model be saved?"
    )
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "done", "next_state"]
    )

    evaluating = False
    epsilon = args.epsilon
    good_tries = 0

    while args.training:
        # Perform episode
        state, done = env.reset(), False
        episode_reward = 0

        while not done:
            if (
                args.render_each
                and env.episode > 0
                and env.episode % args.render_each == 0
            ):
                env.render()

            q_values = network.predict(np.array([state], np.float32))
            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = np.argmax(q_values)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            replay_buffer.append(Transition(state, action, reward, done, next_state))

            if len(replay_buffer) > args.batch_size:

                fastPicker = random.sample(replay_buffer, args.batch_size)
                states = np.array([pick.state for pick in fastPicker], dtype=np.float32)
                actions = np.array([pick.action for pick in fastPicker])
                rewards = np.array(
                    [pick.reward for pick in fastPicker], dtype=np.float32
                )
                next_states = np.array(
                    [pick.next_state for pick in fastPicker], dtype=np.float32
                )
                dones = np.array([pick.done for pick in fastPicker])

                q_values = network.predict(states).numpy()
                next_q = network.predict(next_states).numpy()
                targets = np.copy(q_values)

                for i in range(len(q_values)):
                    targets[i, actions[i]] = rewards[i] + (
                        not dones[i]
                    ) * args.gamma * np.max(next_q[i])

                network.train(states, targets)

            state = next_state
        if episode_reward > 420:
            good_tries += 1
            if good_tries > 20:
                break
        else:
            good_tries = 0

        if args.epsilon_final:
            epsilon = np.exp(
                np.interp(
                    env.episode + 1,
                    [0, args.episodes],
                    [np.log(args.epsilon), np.log(args.epsilon_final)],
                )
            )

    if args.training:
        if args.save:
            network.dump("model")
    else:
        import embedded_data

        network.load("model")

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(network.predict(np.array([state], np.float32))[0])
            state, reward, done, _ = env.step(action)

