import collections
import random
import numpy as np
from utils import preprocess
class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Creates the replay buffer with a fixed maximum size.
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Stores one experience in the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Returns a random batch of experiences.
        """
        # Ensures we don't try to sample more than we have in the buffer
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Separating the experiment tuple into individual batches
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to NumPy arrays
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states),
                np.array(dones))

    def __len__(self):
        """
        Returns the number of stored experiences.
        """
        return len(self.buffer)

    def populate(self, env, policy_func, num_steps, reset_func=None):
        """
        Fills the buffer with data using a given policy.
        """
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        
        for _ in range(num_steps):
            # Get the policy action. We assume that policy_func returns -1, 0, or 1.
            action = policy_func(env,pathfind="astar") 
            
            next_state_raw, reward, done, _ = env.step(action)
            preprocessed_next_state = preprocess(next_state_raw)
            
            # Convert the action to index 0, 1, 2 before adding to the buffer
            action_idx = action + 1 
            self.add(preprocessed_state, action_idx, reward, preprocessed_next_state, done)

            if done:
                state_raw, _, _, _ = env.reset()
                preprocessed_state = preprocess(state_raw)
                if reset_func:
                    reset_func()
            else:
                preprocessed_state = preprocessed_next_state
