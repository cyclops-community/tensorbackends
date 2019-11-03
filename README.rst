TensorBackends
==============

A uniform interface for tensor libraries like NumPy, CTF, etc.

**Notice: This project is under construction. Many functionalities are not
supported yet.**


Backends
--------
- ``numpy``
- ``ctf``


Installation
------------
Considering this package is in development, it is recommended to install it in
the editable mode.

::

    pip install -e /path/to/the/project/directory


Usage
-----
To select a backend,

::

    import tensorbackends as tbs
    tb = tbs.get('numpy')

where ``numpy`` can be replaced with other backend names. If the backend
does not exist or cannot be loaded, an error will be thrown.

Each backend object implements
`the backend interface <tensorbackends/interface/backend.py>`_.

::

    assert isinstance(tb, tbs.interface.Backend)

The tensor object created by a backend implements
`the tensor interface <tensorbackends/interface/tensor.py>`_.

::

    a = tb.empty((2,2), dtype=float)
    assert isinstance(a, tbs.interface.Tensor)

And its type is accessible at ``tb.tensor``.

::

    assert isinstance(a, tb.tensor)
    assert issubclass(tb.tensor, tbs.interface.Tensor)
