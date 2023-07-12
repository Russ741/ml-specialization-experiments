import unittest
import numpy

def f_no_vector(w, b, x):
    y = numpy.full((len(x)), b)
    for x_i in range(len(x)):
        for cell_i in range(len(x[x_i])):
            cell = x[x_i][cell_i]
            featureParameterProduct = cell * w[cell_i]
            y[x_i] += featureParameterProduct
    return y

def f_vector(w, b, x):
    return numpy.dot(x, w) + b

