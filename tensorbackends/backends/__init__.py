"""
This module implements backend selection.
"""


_BACKENDS = {}

def get(name):
    if name not in _BACKENDS:
        raise ValueError(f"Backend {name} does not exsit")
    try:
        return  _BACKENDS[name]()
    except Exception as e:
        raise ValueError(f"Backend {name} cannot be loaded") from e

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
