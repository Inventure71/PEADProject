"""
File for declaring example data
"""

def get_simple_example():
    features = ["yellow", "round", "long"]
    classes = ["banana", "apple"]
    training_data = [
        # yellow, round, long | class
        ([1, 0, 1], "banana"),
        ([1, 0, 1], "banana"),
        ([0, 1, 0], "apple"),
    ]
    test_inputs = [
        ([1, 0, 1], "banana"),
        ([0, 1, 0], "apple"),
    ]

    return features, classes, training_data, test_inputs


def get_complex_example():
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

    test_inputs = [
        ([1, 0, 0, 0, 1, 1, 0, 1], "banana"),
        ([0, 1, 0, 1, 0, 1, 1, 0], "apple"),
        ([0, 0, 1, 0, 1, 1, 0, 1], "pear"),
    ]

    return features, classes, training_data, test_inputs
