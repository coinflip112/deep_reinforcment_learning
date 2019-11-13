import numpy as np
import lunar_lander_evaluator
import embedded_data

if __name__ == "__main__":

    Q1 = np.load("action_value_function_q1.npz", allow_pickle=True)
    Q2 = np.load("action_value_function_q2.npz", allow_pickle=True)
    env = lunar_lander_evaluator.environment()
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False

        while not done:
            action = np.argmax(Q1[state, :] + Q2[state, :])

            next_state, reward, done, _ = env.step(action)
            state = next_state
