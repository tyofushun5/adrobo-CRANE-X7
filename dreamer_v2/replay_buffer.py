from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        image_shape: Tuple[int, int, int],
        joint_dim: int,
        action_dim: int,
    ):
        self.capacity = capacity
        self.image_shape = image_shape
        self.joint_dim = joint_dim
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.images = np.zeros((capacity, *image_shape), dtype=np.float32)
        self.joints = np.zeros((capacity, joint_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        self.images[self.index] = observation["image"]
        self.joints[self.index] = observation["joint_pos"]
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done
        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        episode_ends = np.where(self.done)[0]
        sampled_indexes = []
        max_index = len(self)
        for _ in range(batch_size):
            while True:
                initial_index = np.random.randint(0, max_index - chunk_length)
                final_index = initial_index + chunk_length - 1
                if not np.logical_and(initial_index <= episode_ends, episode_ends < final_index).any():
                    break
            sampled_indexes.extend(range(initial_index, final_index + 1))

        sample_shape = (batch_size, chunk_length)
        sampled_images = self.images[sampled_indexes].reshape(*sample_shape, *self.image_shape)
        sampled_joints = self.joints[sampled_indexes].reshape(*sample_shape, self.joint_dim)
        sampled_actions = self.actions[sampled_indexes].reshape(batch_size, chunk_length, -1)
        sampled_rewards = self.rewards[sampled_indexes].reshape(batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(batch_size, chunk_length, 1)
        observations = {"image": sampled_images, "joint_pos": sampled_joints}
        return observations, sampled_actions, sampled_rewards, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index
