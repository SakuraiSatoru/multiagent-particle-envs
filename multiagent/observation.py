import numpy as np

class Conv1dObservation(object):

    def __init__(self, non_lidar_obs, lidar_obs):
        """

        :type non_lidar_obs: np.ndarray
        :type lidar_obs: np.ndarray
        """
        assert len(non_lidar_obs.shape) == 1
        assert len(lidar_obs.shape) == 2
        self._non_lidar_obs = non_lidar_obs
        self._lidar_obs = lidar_obs

    def __len__(self):
        return self._non_lidar_obs.shape[0] + self._lidar_obs.shape[0]*self._lidar_obs.shape[1]

    @property
    def non_lidar_obs(self):
        return self._non_lidar_obs

    @property
    def lidar_obs(self):
        return self._lidar_obs

    # @property
    # def obs(self):
    #     return self._non_lidar_obs, self._lidar_obs

    @property
    def flatten(self):
        return np.concatenate((self._non_lidar_obs, self._lidar_obs.flatten()), axis=None)

    @property
    def shape(self):
        return self._non_lidar_obs.shape, self._lidar_obs.shape