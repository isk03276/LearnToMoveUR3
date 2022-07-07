from gym import Env, ObservationWrapper
from gym.spaces import Box


class ImageObsWrapper(ObservationWrapper):
    def __init__(self, env: Env, use_arm_camera: bool = True) -> None:
        super().__init__(env)
        self.use_arm_camera = use_arm_camera

    def observation(self, _observation):
        return self.env.render()

    def _define_observation_space(self):
        channel_num = 6 if self.use_arm_camera else 3
        return Box(0, 255, (84, 84, channel_num))
