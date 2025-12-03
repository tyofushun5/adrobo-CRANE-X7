import abc


class Entity(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, position=None):
        if position is None:
            position = [0.0, 0.0, 0.0]
        self.position = position

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        pass

class Robot(Entity):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def action(self, *args, **kwargs):
        pass
