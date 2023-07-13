# To run:
# pytest gradient_descent.py

from numpy.testing import assert_array_equal
from gradient_descent import derivatives_no_vector, derivatives_vector, f_vector, f_no_vector, gradient_descent_no_vectors, j_no_vector, j_vector, slope_no_vector, slope_vector, update_parameters_no_vectors, update_parameters_vectors

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

def test_derivatives():
    slopes = [[-1, 2, 5], [8, 0, 1]]

    expected = [2, 3]
    assert_array_equal(derivatives_no_vector(slopes), expected)
    assert_array_equal(derivatives_vector(slopes), expected)

def test_update_parameters():
    w = [2, 9, 4]
    alpha = 0.001
    derivatives = [100, 0, -3000]

    expected = [1.9, 9, 7]
    assert_array_equal(update_parameters_no_vectors(w, alpha, derivatives), expected)
    assert_array_equal(update_parameters_vectors(w, alpha, derivatives), expected)

def test_gradient_descent():
    w = [1]
    b = 5
    x = [[1], [10]]
    y = [3, 12]

    w_expected = [1]
    b_expected = 2

    w_no_vectors, b_no_vectors = gradient_descent_no_vectors(w, b, x, y, 0.01, 1000)
    assert_array_equal(w_no_vectors, w_expected)
    assert b_no_vectors == b_expected

def test_gradient_descent_lab():
    x = [[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]
    y = [460, 232, 178]

    w = [0, 0, 0, 0]
    b = 0.

    w_expected = [0.2, 0, -0.1, -0.07]
    b_expected = -0.00

    w_no_vectors, b_no_vectors = gradient_descent_no_vectors(w, b, x, y, 5.0e-7, 1000)
    assert_array_equal(w_no_vectors, w_expected)
    assert b_no_vectors == b_expected
