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


Testing
-------
In the project directory, run

::

    python -m unittest test

Then same tests will be run for all backends. However, if a backend other than
``numpy`` is not available, the tests for it will be skipped.


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


Writing tests for multiple backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We provide a convenient way to write test cases for multiple backends
(with ``unittest`` from the Python Standard Library).

In order to do so, users can define test cases that inherit
``unittest.TestCase`` as usual, but add an extra argument (which represents
the backend object) to each "test method" (i.e. methods whose name starts with
"test") and decorate the test case class using
``@test_with_backend(required, optional)`` from ``tensorbackends.utils``,
where ``required`` and ``optional`` are both list of backend names.
Then each test method in this class will be replaced with a set of "instance"
test methods according to the backend names in ``required`` and ``optional``.
For backends listed in ``required``, the instance test method will run in any
case (and fail if the backend is not available). However, for backends listed
in ``optional``, the instance test method will be skipped if the backend is
not available.

Here is an example of a test case class decorated by ``@test_with_backend()``:

::

    import unittest
    from tensorbackends.utils import test_with_backend

    @test_with_backend(['numpy'], optional=['ctf'])
    class SimpleTest(unittest.TestCase):
        def test_shape(self, tb):
            a = tb.empty((2, 3))
            self.assertEqual(a.shape, (2, 3))

And it is roughly equivalant to

::

    import unittest
    import tensorbackends as tbs

    class SimpleTest(unittest.TestCase):
        def test_shape_numpy(self):
            tb = tbs.get('numpy')
            a = tb.empty((2, 3))
            self.assertEqual(a.shape, (2, 3))

        @unittest.skipUnless(tbs.isavailable('ctf'), 'Backend ctf is not available')
        def test_shape_ctf(self):
            tb = tbs.get('ctf')
            a = tb.empty((2, 3))
            self.assertEqual(a.shape, (2, 3))
