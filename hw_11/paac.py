#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import gym_evaluator


class Network:
    def __init__(self, env, args):
        inputs_policy = tf.keras.layers.Input(shape=env.state_shape)
        inputs_baseline = tf.keras.layers.Input(shape=env.state_shape)

        x_policy = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(
            inputs_policy
        )
        x_baseline = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(
            inputs_baseline
        )
        predictions_policy = tf.keras.layers.Dense(env.actions, activation="softmax")(
            x_policy
        )
        predictions_baseline = tf.keras.layers.Dense(1, activation=None)(x_baseline)

        model_policy = tf.keras.models.Model(
            inputs=inputs_policy, outputs=predictions_policy
        )
        model_baseline = tf.keras.models.Model(
            inputs=inputs_baseline, outputs=predictions_baseline
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
        self._policy = model_policy
        self._value = model_baseline

    def train(self, states, actions, returns):
        states, actions, returns = (
            np.array(states, np.float32),
            np.array(actions, np.int32),
            np.array(returns, np.float32),
        )
        baseline_prediction = self._value.predict_on_batch(states)[:, 0]
        centered_returns = returns - baseline_prediction

        self._value.train_on_batch(states, returns)
        self._policy.train_on_batch(states, actions, sample_weight=centered_returns)

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._policy.predict_on_batch(states)

    def predict_values(self, states):
        states = np.array(states, np.float32)
        return self._value.predict_on_batch(states)[:, 0]

    def save(self):
        self._policy.save("paac_policy.h5")
        self._value.save("paac_baseline.h5")


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument(
        "--evaluate_each",
        default=256,
        type=int,
        help="Evaluate each number of batches.",
    )
    parser.add_argument(
        "--evaluate_for", default=100, type=int, help="Evaluate for number of batches."
    )
    parser.add_argument("--gamma", default=0.96, type=float, help="Discounting factor.")
    parser.add_argument(
        "--hidden_layer", default=256, type=int, help="Size of hidden layer."
    )
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--render_each", default=None, type=int, help="Render some episodes."
    )
    parser.add_argument(
        "--threads", default=4, type=int, help="Maximum number of threads to use."
    )
    parser.add_argument(
        "--workers", default=256, type=int, help="Number of parallel workers."
    )
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(env, args)

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            action_probs = network.predict_actions(states)
            actions = np.array(
                [
                    np.random.choice(
                        a=list(range(env.actions)), p=action_probs[row_index, :]
                    )
                    for row_index in range(action_probs.shape[0])
                ]
            )
            step_results = np.array(env.parallel_step(actions))

            next_states = [next_state for next_state, reward, done, _ in step_results]
            done_indicators = [done for next_state, reward, done, _ in step_results]
            rewards = np.array([reward for next_state, reward, done, _ in step_results])

            val_func_pred = network.predict_values(next_states)

            returns = rewards + np.where(
                done_indicators, np.zeros_like(rewards), args.gamma * val_func_pred
            )
            network.train(states, actions, returns)
            states = next_states

        # Periodic evaluation

        returns = []
        for _ in range(args.evaluate_for):
            returns.append(0)
            state, done = env.reset(), False
            while not done:
                if (
                    args.render_each
                    and env.episode > 0
                    and env.episode % args.render_each == 0
                ):
                    env.render()

                probabilities = network.predict_actions([state])[0]
                action = np.argmax(probabilities)
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        mean_return = np.mean(returns)
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, mean_return))
        if mean_return > 490:
            training = False
    network.save()
    for _ in range(100):
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict_actions([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)
