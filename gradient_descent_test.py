# To run:
# pytest gradient_descent.py

from numpy.testing import assert_array_equal
from gradient_descent import f_vector, f_no_vector, j_no_vector, j_vector, slope_no_vector, slope_vector

def test_f():
    w = [2, 3]
    b = 7
    x = [[5, 11], [0, 0], [9, 8]]

    expected = [50, 7, 49]
    assert_array_equal(f_no_vector(w, b, x), expected)
    assert_array_equal(f_vector(w, b, x), expected)

def test_j():
    y_hat = [7, 3, 8]
    y = [9, 3, 5]

    expected = 13 / 6
    assert j_no_vector(y_hat, y) == expected
    assert j_vector(y_hat, y) == expected

def test_slope():
    x = [[5, 11], [0, 0], [9, 8]]
    y_hat = [7, 3, 8]
    y = [9, 3, 5]

    expected = [[-10, 0, 27], [-22, 0, 24]]
    assert_array_equal(slope_no_vector(x, y_hat, y), expected)
    assert_array_equal(slope_vector(x, y_hat, y), expected)
