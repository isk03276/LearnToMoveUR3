from gym import Env, ObservationWrapper
from gym.spaces import Box

from utils.image import resize_image


class ImageObsWrapper(ObservationWrapper):
    def __init__(
        self, env: Env, use_arm_camera: bool = True, width: int = 84, height: int = 84
    ):
        super().__init__(env)
        self.use_arm_camera = use_arm_camera
        self.width = width
        self.height = height
        self._modify_observation_space()

    def observation(self, _observation):
        obs = self.env.render()
        obs = resize_image(obs, self.width, self.height)
        return obs

    def _modify_observation_space(self):
        channel_num = 6 if self.use_arm_camera else 3
        self.env.observation_space = Box(0, 255, (self.width, self.height, channel_num))
