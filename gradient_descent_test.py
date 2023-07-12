# To run:
# pytest gradient_descent.py

from numpy.testing import assert_array_equal
from gradient_descent import f, f_no_vectors

def test_f():
    w = [2, 3]
    b = 7
    x = [[5, 11], [0, 0], [9, 8]]

    expected = [50, 7, 49]
    assert_array_equal(f_no_vectors(w, b, x), expected)
    assert_array_equal(f(w, b, x), expected)