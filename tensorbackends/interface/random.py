"""
This module defines the interface of a random module.
"""

class Random:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def seed(self, seed):
        raise NotImplementedError()

    def rand(self, *dims):
        raise NotImplementedError()
