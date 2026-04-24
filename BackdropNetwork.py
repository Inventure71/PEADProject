import numpy as np
from _ExampleData import get_complex_example, get_simple_example

"""
Features:
yellow -> 0,1
round -> 0,1
long -> 0,1

Classes:
banana
apple


[yellow, round, long] -> (banana, apple)

yellow = 1
round = 0
long = 1
class = banana

[1, 0, 1] -> [1, 0]
"""

run_complex_example = True

epochs = 20
learning_rate = 0.1

features = []
classes = []
training_data = []
test_inputs = []


def softmax(scores):
    # softmax turns raw scores into probabilities.
    # example:
    # raw scores = [2.0, 1.0, 0.5]
    # probabilities might become [0.63, 0.23, 0.14]
    # all probabilities add up to 1.
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def train_backprop_network(epochs, learning_rate, training_data):
    # rows = features, columns = classes
    # same shape as Hebbian weights.
    weights = np.zeros((len(features), len(classes)))

    for epoch in range(epochs):
        print(f"Backprop Epoch {epoch + 1}")

        total_loss = 0

        for inputs, label in training_data:
            inputs = np.array(inputs) # banana: [1, 0, 0, 0, 1, 1, 0, 1]

            # create the correct answer vector.
            # if classes = ["banana", "apple", "pear"]
            # and label = "banana"
            # then target = [1, 0, 0]
            target = np.zeros(len(classes))
            class_index = classes.index(label)
            target[class_index] = 1

            # forward pass:
            # the model uses the current weights to make raw class scores.
            #
            # example:
            # scores[0] = banana score
            # scores[1] = apple score
            # scores[2] = pear score
            scores = np.dot(inputs, weights)

            # turn raw scores into probabilities.
            # example:
            # probabilities = [0.33, 0.33, 0.33] at the beginning
            # because all weights start at zero.
            probabilities = softmax(scores)

            # calculate the error.
            #
            # if correct answer is banana:
            # target        = [1.00, 0.00, 0.00]
            # probabilities = [0.33, 0.33, 0.33]
            #
            # error = probabilities - target
            # error = [-0.67, 0.33, 0.33]
            #
            # meaning:
            # banana is too low, apple and pear are too high.
            error = probabilities - target

            # this is the backprop weight update for this simple model.
            #
            # np.outer(inputs, error) creates a matrix showing how each
            # active feature contributed to each class error.
            #
            # then we subtract it because we want to reduce the error.
            weights -= learning_rate * np.outer(inputs, error)

            # simple loss number.
            # lower loss means the model is becoming more confident
            # in the correct class.
            loss = -np.log(probabilities[class_index] + 1e-9)
            total_loss += loss

        print("Loss:", total_loss)
        print(weights)

    print("--------------------------------")
    print("Final backprop weights:")
    print(weights)

    return weights


def predict(weights, inputs):
    # use the learned weights to make class scores.
    # example: scores = [2.1, 0.4, 1.2]
    # the biggest score is the predicted class.
    inputs = np.array(inputs)
    scores = np.dot(inputs, weights)
    return classes[np.argmax(scores)]


def run_example(example_loader):
    global features, classes, training_data, test_inputs
    features, classes, training_data, test_inputs = example_loader()

    weights = train_backprop_network(epochs, learning_rate, training_data)

    print("\nPredictions:")
    for inputs, expected_label in test_inputs:
        prediction = predict(weights, inputs)
        print(f"{prediction}  # should be {expected_label}")

    return weights


if run_complex_example:
    weights = run_example(get_complex_example)
else:
    weights = run_example(get_simple_example)
