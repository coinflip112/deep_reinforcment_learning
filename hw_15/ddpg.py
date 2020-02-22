#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf

import gym_evaluator


class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]
        inputs_actor = tf.keras.layers.Input(shape=env.state_shape)
        hidden_actor = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(
            inputs_actor
        )
        hidden_actor = tf.keras.layers.Dense(
            int(args.hidden_layer / 2), activation="relu"
        )(hidden_actor)
        output_actor = tf.keras.layers.Dense(action_components, activation="tanh")(
            hidden_actor
        )
        self._actor = tf.keras.models.Model(
            inputs=[inputs_actor], outputs=[output_actor]
        )
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        self._actor.compile(
            optimizer=self.actor_opt,
            loss=tf.losses.MeanSquaredError(),
            experimental_run_tf_function=False,
        )

        self._target_actor = tf.keras.models.clone_model(self._actor)

        inputs_critic = tf.keras.layers.Input(shape=4)
        hidden_critic = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(
            inputs_critic
        )
        hidden_critic = tf.keras.layers.Concatenate()([inputs_actor, hidden_critic])
        hidden_critic = tf.keras.layers.Dense(
            int(args.hidden_layer / 2), activation="relu"
        )(hidden_critic)
        output_critic = tf.keras.layers.Dense(1, activation=None)(hidden_critic)

        self._critic = tf.keras.models.Model(
            inputs=[inputs_actor, inputs_critic], outputs=output_critic
        )
        self._critic.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            experimental_run_tf_function=False,
        )

        self._target_critic = tf.keras.models.clone_model(self._critic)
        self.tau = args.target_tau

    @tf.function
    def _train(self, states, actions, returns):
        with tf.GradientTape() as tape:
            actions = self._actor(states)
            tape.watch(actions)
            q_values = self._critic([states, actions])
        dq_da = tape.gradient(q_values, actions)

        with tf.GradientTape() as tape:
            actions = self._actor(states)
            theta = self._actor.trainable_variables
        da_dtheta = tape.gradient(actions, theta, output_gradients=-dq_da)
        self.actor_opt.apply_gradients(zip(da_dtheta, self._actor.trainable_variables))

    def train(self, states, actions, returns):
        states, actions, returns = (
            np.array(states, np.float32),
            np.array(actions, np.float32),
            np.array(returns, np.float32),
        )
        self._critic.train_on_batch((states, actions), returns)

        self._train(states, actions, returns)

        self._target_actor.set_weights(
            (1 - self.tau) * np.array(self._target_actor.get_weights())
            + self.tau * np.array(self._actor.get_weights())
        )
        self._target_critic.set_weights(
            (1 - self.tau) * np.array(self._target_critic.get_weights())
            + self.tau * np.array(self._critic.get_weights())
        )

    @tf.function
    def _predict_actions(self, states):
        return self._actor(states, training=False)

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._predict_actions(states).numpy()

    @tf.function
    def _predict_values(self, states):
        actions = self._target_actor(states, training=False)
        return self._target_critic((states, actions), training=False)

    def predict_values(self, states):
        states = np.array(states, np.float32)
        return self._predict_values(states).numpy()[:, 0]


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(
            scale=self.sigma, size=self.state.shape
        )
        return self.state


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument(
        "--env", default="BipedalWalker-v2", type=str, help="Environment."
    )
    parser.add_argument(
        "--evaluate_each",
        default=100,
        type=int,
        help="Evaluate each number of episodes.",
    )
    parser.add_argument(
        "--evaluate_for", default=50, type=int, help="Evaluate for number of batches."
    )
    parser.add_argument(
        "--noise_sigma", default=0.3, type=float, help="UB noise sigma."
    )
    parser.add_argument(
        "--noise_theta", default=0.2, type=float, help="UB noise theta."
    )
    parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
    parser.add_argument(
        "--hidden_layer", default=256, type=int, help="Size of hidden layer."
    )
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--render_each", default=100, type=int, help="Render some episodes."
    )
    parser.add_argument(
        "--target_tau", default=0.005, type=float, help="Target network update weight."
    )
    parser.add_argument(
        "--threads", default=8, type=int, help="Maximum number of threads to use."
    )
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(128)
    tf.random.set_seed(128)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)
    action_lows, action_highs = map(np.array, env.action_ranges)

    # Construct the network
    network = Network(env, args)

    replay_buffer = collections.deque(maxlen=int(1e6))
    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "done", "next_state"]
    )

    noise = OrnsteinUhlenbeckNoise(
        env.action_shape[0], 0.0, args.noise_theta, args.noise_sigma
    )
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                action = np.clip(
                    network.predict_actions([state])[0] + noise.sample(), [-1.0], [1.0]
                )
                next_state, reward, done, _ = env.step(action)

                replay_buffer.append(
                    Transition(state, action, reward, done, next_state)
                )
                state = next_state

                if len(replay_buffer) >= int(1e4):
                    batch = np.random.choice(
                        len(replay_buffer), size=args.batch_size, replace=False
                    )
                    states, actions, rewards, dones, next_states = zip(
                        *[replay_buffer[i] for i in batch]
                    )
                    predicted = network.predict_values(next_states)
                    returns = rewards + args.gamma * np.array(
                        [
                            (predicted[idx] if not dones[idx] else 0)
                            for idx in range(args.batch_size)
                        ]
                    )
                    network.train(states, actions, returns)

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

                action = network.predict_actions([state])[0]
                state, reward, done, _ = env.step(action)
                returns[-1] += reward
        print(
            "Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns))
        )
        if np.mean(returns) >= 110:
            training = False
        env._env.close()
    for _ in range(100):
        state, done = env.reset(True), False
        while not done:
            action = network.predict_actions([state])[0]
            state, reward, done, _ = env.step(action)
