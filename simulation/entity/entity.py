import abc


class Entity(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, scene = None, surface=None):
        self.scene = scene
        self.surface = surface

    @abc.abstractmethod
    def create(self):
        pass

