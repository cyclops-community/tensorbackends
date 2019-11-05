

def mimic_operators(attr):
    def add_unary_operator(cls, operator_name):
        def method(self):
            return getattr(getattr(self, attr), operator_name)()
        method.__module__ = cls.__module__
        method.__qualname__ = f'{cls.__qualname__}.{operator_name}'
        method.__name__ = operator_name
        setattr(cls, operator_name, method)

    def add_binary_operator(cls, operator_name):
        def method(self, other):
            if isinstance(other, cls):
                other = getattr(other, attr)
            return getattr(getattr(self, attr), operator_name)(other)
        method.__module__ = cls.__module__
        method.__qualname__ = f'{cls.__qualname__}.{operator_name}'
        method.__name__ = operator_name
        setattr(cls, operator_name, method)

    def decorator(cls):
        add_unary_operator(cls, '__pos__')
        add_unary_operator(cls, '__neg__')
        add_unary_operator(cls, '__abs__')
    
        add_binary_operator(cls, '__add__')
        add_binary_operator(cls, '__sub__')
        add_binary_operator(cls, '__mul__')
        add_binary_operator(cls, '__matmul__')
        add_binary_operator(cls, '__truediv__')
        add_binary_operator(cls, '__floordiv__')
        add_binary_operator(cls, '__pow__')

        add_binary_operator(cls, '__radd__')
        add_binary_operator(cls, '__rsub__')
        add_binary_operator(cls, '__rmul__')
        add_binary_operator(cls, '__rmatmul__')
        add_binary_operator(cls, '__rtruediv__')
        add_binary_operator(cls, '__rfloordiv__')
        add_binary_operator(cls, '__rpow__')

        add_binary_operator(cls, '__iadd__')
        add_binary_operator(cls, '__isub__')
        add_binary_operator(cls, '__imul__')
        add_binary_operator(cls, '__imatmul__')
        add_binary_operator(cls, '__itruediv__')
        add_binary_operator(cls, '__ifloordiv__')
        add_binary_operator(cls, '__ipow__')
        return cls

    return decorator
