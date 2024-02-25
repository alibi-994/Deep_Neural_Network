from deep_neural_network_modules import * 


def dnn_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 1000):
    parameters = initialize_parameters(layer_dims)

    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, caches, parameters)
        update_parameters(parameters, grads, learning_rate=learning_rate)
    
    return parameters
