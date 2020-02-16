#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf

import gym_evaluator


class Network:
    def __init__(self, env, args):
        assert len(env.action_shape) == 1
        action_components = env.action_shape[0]
        action_lows, action_highs = map(np.array, env.action_ranges)
        action_diffs = np.array(
            action_highs[i] - action_lows[i] for i in range(len(actions))
        )

        inputs = tf.keras.layers.Input(shape=env.weights)
        x = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(inputs)
        x = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(x)
        output = tf.keras.layers.Dense(action_components, activation=tf.nn.sigmoid())(x)
        output = tf.keras.layers.multiply()([action_diffs, output])
        output = tf.keras.layers.add()([action_lows, output])
        self._actor = tf.keras.models.Model(inputs=[inputs], outputs=[output])
        self._actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError,
            experimental_run_tf_function=False,
        )

        self._target_actor = tf.keras.models.clone_model(self._actor)

        input_actions = tf.keras.layers.Input(shape=1)
        x = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(inputs)
        x = tf.keras.layers.Concatenate()([x, input_actions])
        x = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(x)
        x = tf.keras.layers.Dense(args.hidden_layer, activation="relu")(x)
        output_critic = tf.keras.layers.Dense(1, activation=None)(x)

        self._critic = tf.keras.models.Model(
            inputs=[inputs, input_actions], outputs=output_critic
        )
        self._critic.compile(
            loss=tf.keras.losses.MeanSquaredError,
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        )

        self._target_critic = tf.keras.models.clone_model(self._critic)

    @tf.function
    def _train(self, states, actions, returns):
        with tf.GradientTape() as tape:
            self._critic().train_on_batch([states,returns)
            pass
        # TODO: Train separately the actor and critic.
        #
        # Furthermore, update the weights of the target actor and critic networks
        # by using args.target_tau option.
        pass

    def train(self, states, actions, returns):
        states, actions, returns = (
            np.array(states, np.float32),
            np.array(actions, np.float32),
            np.array(returns, np.float32),
        )
        self._train(states, actions, returns)

    @tf.function
    def _predict_actions(self, states):
        # TODO: Compute actions by the actor
        pass

    def predict_actions(self, states):
        states = np.array(states, np.float32)
        return self._predict_actions(states).numpy()

    @tf.function
    def _predict_values(self, states):
        # TODO: Predict actions by the target actor and evaluate them using
        # target_critic.
        pass

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
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
    parser.add_argument(
        "--evaluate_each",
        default=50,
        type=int,
        help="Evaluate each number of episodes.",
    )
    parser.add_argument(
        "--evaluate_for", default=10, type=int, help="Evaluate for number of batches."
    )
    parser.add_argument(
        "--noise_sigma", default=0.2, type=float, help="UB noise sigma."
    )
    parser.add_argument(
        "--noise_theta", default=0.15, type=float, help="UB noise theta."
    )
    parser.add_argument("--gamma", default=None, type=float, help="Discounting factor.")
    parser.add_argument(
        "--hidden_layer", default=None, type=int, help="Size of hidden layer."
    )
    parser.add_argument(
        "--learning_rate", default=None, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--render_each", default=0, type=int, help="Render some episodes."
    )
    parser.add_argument(
        "--target_tau", default=None, type=float, help="Target network update weight."
    )
    parser.add_argument(
        "--threads", default=1, type=int, help="Maximum number of threads to use."
    )
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)
    action_lows, action_highs = map(np.array, env.action_ranges)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "done", "next_state"]
    )

    noise = OrnsteinUhlenbeckNoise(
        env.action_shape[0], 0.0, args.noise_theta, args.noise_sigma
    )
    while True:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                # TODO: Perform an action and store the transition in the replay buffer

                # If the replay_buffer is large enough, perform training
                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(
                        len(replay_buffer), size=args.batch_size, replace=False
                    )
                    states, actions, rewards, dones, next_states = zip(
                        *[replay_buffer[i] for i in batch]
                    )
                    # TODO: Perform the training

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

    # On the end perform final evaluations with `env.reset(True)`
