"""
This module implements test utilities.
"""

import functools, inspect, unittest


def test_with_backend(required=['numpy'], optional=['ctf', 'ctfview']):
    from .. import backends
    def instantiate_test_method(name, method, tb_name):
        new_name = '{}_{}'.format(name, tb_name)
        @functools.wraps(method)
        def new_method(self):
            tb = backends.get(tb_name)
            return method(self, tb)
        new_method.__name__ = new_name
        return new_name, new_method
    def decrator(cls):
        test_methods = [
            (name, method)
            for name, method in inspect.getmembers(cls, inspect.isfunction)
            if name.startswith('test')
        ]
        for name, method in test_methods:
            delattr(cls, name)
            for tb_name in required:
                new_name, new_method = instantiate_test_method(name, method, tb_name)
                setattr(cls, new_name, new_method)
            for tb_name in optional:
                new_name, new_method = instantiate_test_method(name, method, tb_name)
                if not backends.isavailable(tb_name):
                    new_method = unittest.skip('Backend {} is not availabe'.format(tb_name))(new_method)
                setattr(cls, new_name, new_method)
        return cls
    return decrator
