import numpy as np

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

epochs = 5
learning_rate = 0.1

features = []
classes = []
training_data = []

def train_hebbian_network(epochs, learning_rate, training_data):
    # Rows = features, columns = classes
    weights = np.zeros((len(features), len(classes)))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        for inputs, label in training_data:
            inputs = np.array(inputs) # [1, 0, 1]

            output = np.zeros(len(classes)) # [0, 0]
            class_index = classes.index(label) # here we find the class index
            output[class_index] = 1 # here we set the value at the class index to 1 (identified)

            # Example for banana:
            # inputs = [1, 0, 1], output = [1, 0]
            # np.outer(inputs, output) =
            # [[1, 0],
            #  [0, 0],
            #  [1, 0]]
            # So yellow->banana and long->banana increase by learning_rate.
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

def simple_example():
    global features, classes, training_data # just to be sure
    features = ["yellow", "round", "long"]
    classes = ["banana", "apple"]
    training_data = [
        ([1, 0, 1], "banana"),
        ([1, 0, 1], "banana"),
        ([0, 1, 0], "apple"),
    ]

    weights = train_hebbian_network(epochs, learning_rate, training_data)

    print(predict(weights, [1, 0, 1])) # should be banana
    print(predict(weights, [0, 1, 0])) # should be apple

    return weights

def complex_example():
    global features, classes, training_data # just to be sure

    features = [
        "yellow",
        "red",
        "green",
        "round",
        "long",
        "sweet",
        "crunchy",
        "soft",
    ]

    classes = [
        "banana",
        "apple",
        "pear",
    ]

    training_data = [
        # yellow, red, green, round, long, sweet, crunchy, soft | class

        ([1, 0, 0, 0, 1, 1, 0, 1], "banana"),
        ([1, 0, 0, 0, 1, 1, 0, 1], "banana"),
        ([1, 0, 0, 0, 1, 1, 0, 1], "banana"),

        ([0, 1, 0, 1, 0, 1, 1, 0], "apple"),
        ([0, 1, 0, 1, 0, 1, 1, 0], "apple"),
        ([0, 0, 1, 1, 0, 1, 1, 0], "apple"),  # green apple

        ([0, 0, 1, 0, 1, 1, 0, 1], "pear"),
        ([0, 0, 1, 0, 1, 1, 0, 1], "pear"),
        ([1, 0, 1, 0, 1, 1, 0, 1], "pear"),  # yellow-green pear
    ]

    weights = train_hebbian_network(epochs, learning_rate, training_data)

    print(predict(weights, [1, 0, 0, 0, 1, 1, 0, 1])) # should be banana
    print(predict(weights, [0, 1, 0, 1, 0, 1, 1, 0])) # should be apple
    print(predict(weights, [0, 0, 1, 0, 1, 1, 0, 1])) # should be pear

    return weights

if run_complex_example:
    weights = complex_example()
else:
    weights = simple_example()