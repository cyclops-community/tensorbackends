"""
This module defines the interface of a backend.
"""

class Backend:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def tensor(self):
        raise NotImplementedError()

    def astensor(self, obj, dtype=None):
        raise NotImplementedError()

    def empty(self, shape, dtype=float):
        raise NotImplementedError()

    def zeros(self, shape, dtype=float):
        raise NotImplementedError()

    def ones(self, shape, dtype=float):
        raise NotImplementedError()

    def copy(self, a):
        raise NotImplementedError()

    def einsvd(self, einstr, a):
        raise NotImplementedError()
