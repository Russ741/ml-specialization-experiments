import numpy

# returns y_hat where y_hat[i] is the predicted value of f_w,b(x[i])
def f_no_vector(w, b, x):
    y = [b] * len(x)
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
    cost = 0.
    m = len(y)
    for i in range(m):
        cost += (y_hat[i] - y[i]) ** 2
    cost *= 1 / (2 * m)
    return cost

def j_vector(y_hat, y):
    return 1 / (2 * len(y)) * numpy.sum(numpy.square(numpy.subtract(y_hat, y)))

# returns an array of slopes where slope[i] is a vector of slopes for parameter i
# slope[param_num][training_sample]
def slope_no_vector(x, y_hat, y):
    m = len(x[0])
    result = []
    for param_num in range(m):
        resultRow = []
        for training_sample in range(len(x)):
            resultRow.append((y_hat[training_sample] - y[training_sample]) * x[training_sample][param_num])
        result.append(resultRow)
    return result

def slope_vector(x, y_hat, y):
    deltas = numpy.subtract(y_hat, y)
    return numpy.transpose(x) * deltas

def derivatives_no_vector(slopes):
    results = []
    training_samples = len(slopes[0])
    for parameter in range(len(slopes)):
        result = 0.
        for training_example in range(training_samples):
            result += slopes[parameter][training_example]
        result /= training_samples
        results.append(result)
    return results

def derivatives_vector(slopes):
    return numpy.sum(slopes, 1) / numpy.shape(slopes)[1]

def update_parameters_no_vectors(w, alpha, derivatives):
    results = []
    for i in range(len(w)):
        results.append(w[i] - alpha * derivatives[i])
    return results

def update_parameters_vectors(w, alpha, derivatives):
    return numpy.subtract(w, numpy.multiply(alpha, derivatives))

def derivative_b_no_vectors(y_hat, y):
    derivative_b = 0
    m = len(y_hat)
    for i in range(m):
        derivative_b += y_hat[i] - y[i]
    return derivative_b / m

def gradient_descent_no_vectors(w, b, x, y, alpha, rounds):
    print(f"Cost before: {j_no_vector(f_no_vector(w, b, x), y)}")
    for i in range(rounds):
        y_hat = f_no_vector(w, b, x)
        # j = j_no_vector(y_hat, y)
        slopes = slope_no_vector(x, y_hat, y)
        derivatives = derivatives_no_vector(slopes)
        new_w = update_parameters_no_vectors(w, alpha, derivatives)
        w = new_w
        derivative_b = derivative_b_no_vectors(y_hat, y)
        new_b = b - alpha * derivative_b
        b = new_b
    #    print(f"round {i}: w = {w}, b = {b}")
    print(f"Cost after: {j_no_vector(f_no_vector(new_w, new_b, x), y)}")
    return new_w, new_b
