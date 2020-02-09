#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import pandas as pd
import continuous_mountain_car_evaluator

# This class is a bare version of tfp.distributions.Normal
class Normal:
    def __init__(self, loc, scale):
        self.loc = tf.convert_to_tensor(loc, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

    def log_prob(self, x):
        log_unnormalized = -0.5 * tf.math.squared_difference(
            x / self.scale, self.loc / self.scale
        )
        log_normalization = 0.5 * np.log(2.0 * np.pi) + tf.math.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * np.log(2.0 * np.pi) + tf.math.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * tf.ones_like(self.loc)

    def sample_n(self, n, seed=None):
        shape = tf.concat(
            [[n], tf.broadcast_dynamic_shape(tf.shape(self.loc), tf.shape(self.scale))],
            axis=0,
        )
        sampled = tf.random.normal(
            shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=seed
        )
        return sampled * self.scale + self.loc


class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]

        self.entropy_regularization = args.entropy_regularization

        inputs = tf.keras.layers.Input(shape=env.weights)
        mus = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(inputs)
        mus = tf.keras.layers.Dense(action_components, activation="tanh")(mus)

        sds = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(inputs)
        sds = tf.keras.layers.Dense(action_components, activation=tf.nn.softplus)(sds)

        values = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(inputs)
        values = tf.keras.layers.Dense(1, activation=None)(values)

        self._model = tf.keras.Model(inputs=inputs, outputs=[mus, sds, values])
        self._optimizer = tf.optimizers.Adam(args.learning_rate)

    @tf.function
    def _train(self, states, actions, returns):
        with tf.GradientTape() as tape:
            mus, sds, values = self._model(states, training=True)
            action_distribution = Normal(mus, sds)
            nll = tf.reduce_sum(-action_distribution.log_prob(actions), axis=1)
            weights = returns - tf.stop_gradient(values)
            loss_1 = tf.math.reduce_mean(nll * weights)
            loss_2 = -action_distribution.entropy() * self.entropy_regularization
            loss_3 = tf.keras.losses.MSE(returns, values)

            loss = loss_1 + loss_2 + loss_3

        gradient = tape.gradient(loss, self._model.variables)
        self._optimizer.apply_gradients(zip(gradient, self._model.variables))

    def train(self, states, actions, returns):
        states, actions, returns = (
            np.array(states, np.int32),
            np.array(actions, np.float32),
            np.array(returns, np.float32),
        )
        self._train(states, actions, returns)

    @tf.function
    def _predict(self, states):
        return self._model(states, training=False)

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        mus, sds, _ = self._predict([states])
        return mus.numpy(), sds.numpy()

    def predict_values(self, states):
        states = np.array(states, np.float32)
        _, _, values = self._predict(states)
        return values.numpy()[:, 0]


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entropy_regularization",
        default=0.06,
        type=float,
        help="Entropy regularization weight.",
    )
    parser.add_argument(
        "--evaluate_each",
        default=1000,
        type=int,
        help="Evaluate each number of batches.",
    )
    parser.add_argument(
        "--evaluate_for", default=10, type=int, help="Evaluate for number of batches."
    )
    parser.add_argument("--gamma", default=0.96, type=float, help="Discounting factor.")
    parser.add_argument(
        "--hidden_layer", default=64, type=int, help="Size of hidden layer."
    )
    parser.add_argument(
        "--learning_rate", default=0.0005, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--render_each", default=None, type=int, help="Render some episodes."
    )
    parser.add_argument(
        "--threads", default=4, type=int, help="Maximum number of threads to use."
    )
    parser.add_argument("--tiles", default=32, type=int, help="Tiles to use.")
    parser.add_argument(
        "--workers", default=128, type=int, help="Number of parallel workers."
    )
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = continuous_mountain_car_evaluator.environment(tiles=args.tiles)
    action_lows, action_highs = env.action_ranges

    # Construct the network
    network = Network(env, args)

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    states = tf.math.reduce_sum(
        tf.one_hot(states, env.weights, axis=2, dtype=tf.int32), axis=1
    )
    training = True
    while training:
        # Training
        for i in range(args.evaluate_each):
            mus, sds = network.predict_actions(states)
            actions = np.array(
                [
                    np.clip(np.random.normal(mu, sd), action_lows, action_highs)
                    for mu, sd in zip(mus, sds)
                ]
            )
            step_results = env.parallel_step(actions)
            next_states = np.array(
                [next_state for next_state, reward, done, _ in step_results]
            )
            done_indicators = [done for next_state, reward, done, _ in step_results]
            rewards = np.array([reward for next_state, reward, done, _ in step_results])
            next_states = tf.math.reduce_sum(
                tf.one_hot(next_states, env.weights, axis=2, dtype=tf.int32), axis=1
            )
            val_func_pred = network.predict_values(next_states)
            returns = rewards + np.where(
                done_indicators, np.zeros_like(rewards), args.gamma * val_func_pred
            )
            network.train(states, actions, returns)
            states = next_states

        # Periodic evaluation
        returns = []
        for j in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if (
                    args.render_each
                    and env.episode > 0
                    and env.episode % args.render_each == 0
                ):
                    env.render()
                state = np.array([1 if i in state else 0 for i in range(env.weights)])
                action = network.predict_actions([state])[0][0]
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        mean_return = np.mean(returns)
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, mean_return))
        if mean_return > 91:
            training = False

    for _ in range(100):
        state, done = env.reset(True), False
        while not done:
            state = np.array([1 if i in state else 0 for i in range(env.weights)])
            action = network.predict_actions([state])[0][0]
            state, reward, done, _ = env.step(action)
