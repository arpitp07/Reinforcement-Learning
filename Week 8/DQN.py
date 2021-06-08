from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
import random
import numpy as np
import pandas as pd


class DQNAgent_1NN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.DISCOUNT = 0.99  # discount rate
        self.epsilon = 0.05  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, day):
        self.memory.append((state, action, reward, next_state, day))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = pd.DataFrame(
            random.sample(self.memory, batch_size),
            columns=["state", "action", "reward", "next_state", "day"],
        )
        state_batch = np.array([x[0] for x in minibatch["state"]])
        next_state = np.array([x[0] for x in minibatch["next_state"]])
        target = minibatch["reward"] + self.DISCOUNT * np.max(
            self.model.predict(next_state), axis=1
        )
        target_f = self.model.predict(state_batch)
        # for x in range(len(target_f)):
        #     target_f[x, minibatch['action'][x]] = target[x]
        target_f[range(len(target_f)), minibatch["action"]] = target
        self.model.fit(state_batch, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class DQNAgent_2NN:
    def __init__(self, state_size, action_size, REPLAY_MEMORY_SIZE=50000):

        self.state_size = state_size
        self.action_size = action_size
        self.target_update_counter = 0
        self.learning_rate = 0.001
        # Main model
        self.model = self.create_model()
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights

    def create_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Adds step's data to a memory replay array
    # (state, action, reward, new state, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # def update_replay_memory(self, state, action, reward, next_state, done):
    # self.replay_memory.append((state, action, reward, next_state, done))

    # Trains main network every step during episode
    def train(
        self,
        terminal_state,
        step,
        MIN_REPLAY_MEMORY_SIZE=1000,
        MINIBATCH_SIZE=64,
        DISCOUNT=0.99,
        UPDATE_TARGET_EVERY=5
    ):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.reshape(state, [1, self.state_size]))[0]

    def load(self, name_base, name_target):
        self.model.load_weights(name_base)
        self.target_model.load_weights(name_target)

    def save(self, name_base, name_target):
        self.model.save_weights(name_base)
        self.target_model.save_weights(name_target)
