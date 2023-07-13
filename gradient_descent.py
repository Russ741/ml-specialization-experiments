import numpy

# returns y_hat where y_hat[i] is the predicted value of f_w,b(x[i])
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

# returns a scalar representing the cost (mean of squares of errors)
def j_no_vector(y_hat, y):
    cost = 0
    m = len(y)
    for i in range(m):
        cost += (y_hat[i] - y[i]) ** 2
    cost *= 1 / (2 * m)
    return cost

def j_vector(y_hat, y):
    return 1 / (2 * len(y)) * numpy.sum(numpy.square(numpy.subtract(y_hat, y)))

