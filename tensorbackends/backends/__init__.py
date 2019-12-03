"""
This module implements backend selection.
"""

from ..interface import Backend

_BACKENDS = {}

def get(obj):
    if isinstance(obj, Backend):
        return obj
    else:
        return get_by_name(obj)

def get_by_name(name):
    if not isinstance(name, str):
        raise TypeError('backend name should be a string, but {} is {}'.format(name, type(name).__qualname__))
    if name not in _BACKENDS:
        raise ValueError('backend {} does not exsit'.format(name))
    try:
        return _BACKENDS[name]()
    except Exception as e:
        raise ValueError('backend {} cannot be loaded'.format(name)) from e

def isavailable(name):
    try:
        get(name)
        return True
    except:
        return False

def register(name):
    def decorator(func):
        _BACKENDS[name] = func
        return None
    return decorator

@register('numpy')
def _():
    from .numpy import NumPyBackend
    return NumPyBackend()

@register('ctf')
def _():
    from .ctf import CTFBackend
    return CTFBackend()

@register('ctfview')
def _():
    from .ctfview import CTFViewBackend
    return CTFViewBackend()
