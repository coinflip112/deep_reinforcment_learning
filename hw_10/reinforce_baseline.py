#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_evaluator


class Network:
    def __init__(self, env, args):
        inputs = tf.keras.layers.Input(shape=env.state_shape)

        for i in range(args.hidden_layers):
            if i == 0:
                x_common = tf.keras.layers.Dense(
                    args.hidden_layer_size, activation="relu"
                )(inputs)
            elif i < 2:
                x_common = tf.keras.layers.Dense(
                    args.hidden_layer_size, activation="relu"
                )(x_common)
            elif i == 2:
                x_policy = tf.keras.layers.Dense(
                    args.hidden_layer_size, activation="relu"
                )(x_common)
                x_baseline = tf.keras.layers.Dense(
                    args.hidden_layer_size, activation="relu"
                )(x_common)
            else:
                x_policy = tf.keras.layers.Dense(
                    args.hidden_layer_size, activation="relu"
                )(x_policy)
                x_baseline = tf.keras.layers.Dense(
                    args.hidden_layer_size, activation="relu"
                )(x_baseline)

        predictions_policy = tf.keras.layers.Dense(env.actions, activation="softmax")(
            x_policy
        )
        predictions_baseline = tf.keras.layers.Dense(1, activation=None)(x_baseline)

        model_policy = tf.keras.models.Model(inputs=inputs, outputs=predictions_policy)
        model_baseline = tf.keras.models.Model(
            inputs=inputs, outputs=predictions_baseline
        )

        model_policy.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            experimental_run_tf_function=False,
        )

        model_baseline.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            experimental_run_tf_function=False,
        )
        self.model = model_policy
        self.baseline = model_baseline

    def train(self, states, actions, returns):
        states = [val for sublist in states for val in sublist]
        actions = [val for sublist in actions for val in sublist]
        returns = [val for sublist in returns for val in sublist]
        states, actions, returns = (
            np.array(states, np.float32),
            np.array(actions, np.int32),
            np.array(returns, np.float32),
        )
        baseline_prediction = self.baseline.predict_on_batch(states)
        centered_returns = returns - baseline_prediction.squeeze()

        self.model.train_on_batch(states, actions, sample_weight=centered_returns)
        self.baseline.train_on_batch(states, returns)

    def predict(self, states):
        states = np.array(states, np.float32)

        return self.model.predict(states)

    def save(self):
        self.model.save("reinforce_policy.h5")
        self.baseline.save("reinforce_baseline.h5")


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Number of episodes to train on."
    )
    parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=0.98, type=float, help="Discounting factor.")
    parser.add_argument(
        "--hidden_layers", default=5, type=int, help="Number of hidden layers."
    )
    parser.add_argument(
        "--hidden_layer_size", default=16, type=int, help="Size of hidden layer."
    )
    parser.add_argument(
        "--learning_rate", default=0.01, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--render_each", default=None, type=int, help="Render some episodes."
    )
    parser.add_argument(
        "--threads", default=4, type=int, help="Maximum number of threads to use."
    )
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if (
                    args.render_each
                    and env.episode > 0
                    and env.episode % args.render_each == 0
                ):
                    env.render()

                action_probs = network.predict([state])[0]

                action = np.random.choice(list(range(env.actions)), p=action_probs)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns by summing rewards (with discounting)

            single_return = 0
            returns = []

            for reward in np.flip(rewards):
                single_return = reward + args.gamma * single_return
                returns.append(single_return)

            batch_states.append(np.array(states))
            batch_actions.append(np.array(actions))
            batch_returns.append(np.flip(returns))

        # Train using the generated batch
        network.train(batch_states, batch_actions, batch_returns)
    network.save()
    # Final evaluation
    for _ in range(100):
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
