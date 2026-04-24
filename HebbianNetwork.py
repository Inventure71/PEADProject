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

epochs = 5
learning_rate = 0.1

features = ["yellow", "round", "long"]
classes = ["banana", "apple"]

training_data = [
    # features: yellow, round, long | class
    ([1, 0, 1], "banana"),
    ([1, 0, 1], "banana"),
    ([0, 1, 0], "apple"),
    ([0, 1, 0], "apple"),
]

# Rows = features, columns = classes
weights = np.zeros((3, 2))

def train_hebbian_network(epochs, learning_rate, training_data):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        for inputs, label in training_data:
            inputs = np.array(inputs) # [1, 0, 1]

            output = np.zeros(len(classes)) # [0, 0]
            class_index = classes.index(label) # here we find the class index
            output[class_index] = 1 # here we set the value at the class index to 1 (identified)

            weights += learning_rate * np.outer(inputs, output) #{insert math} 

        print(weights)

    print("--------------------------------")
    print("Final weights:")
    print(weights)
    return weights

weights = train_hebbian_network(epochs, learning_rate, training_data)