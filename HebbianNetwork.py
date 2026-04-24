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

epochs = 1
learning_rate = 0.1

features = []
classes = []
training_data = []
test_inputs = []

def train_hebbian_network(epochs, learning_rate, training_data):
    # rows = features, columns = classes
    weights = np.zeros((len(features), len(classes)))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        for inputs, label in training_data:
            inputs = np.array(inputs) # [1, 0, 1]

            output = np.zeros(len(classes)) # [0, 0]
            class_index = classes.index(label) # here we find the class index
            output[class_index] = 1 # here we set the value at the class index to 1 (identified)

            # example for banana:
            # inputs = [1, 0, 1], output = [1, 0]
            # np.outer(inputs, output) =
            # [[1, 0],
            #  [0, 0],
            #  [1, 0]]
            # so yellow->banana and long->banana increase by learning_rate.
            weights += learning_rate * np.outer(inputs, output)

        print(weights)

    print("--------------------------------")
    print("Final weights:")
    print(weights)
    return weights

def predict(weights, inputs):
    inputs = np.array(inputs)
    output = np.dot(inputs, weights) # one score per class
    return classes[np.argmax(output)]

def run_example(example_loader):
    global features, classes, training_data, test_inputs
    features, classes, training_data, test_inputs = example_loader()

    weights = train_hebbian_network(epochs, learning_rate, training_data)

    print("\nPredictions:")
    for inputs, expected_label in test_inputs:
        prediction = predict(weights, inputs)
        print(f"{prediction}  # should be {expected_label}")

    return weights


if run_complex_example:
    weights = run_example(get_complex_example)
else:
    weights = run_example(get_simple_example)
