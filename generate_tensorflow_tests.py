import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sinode"))
)

import sinode.sinode as sinode
import tensorflow as tf

input = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
input = tf.reshape(input, [1, 3, 3, 1])


class Trial(sinode.Sinode):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)
        self.operator = self.function(**self.args)
        self.result = self.operator(self.input)
        self.input = input.numpy()
        self.result = self.result.numpy()


trials = [
    Trial(
        input=input,
        function=tf.keras.layers.MaxPool2D,
        args={"pool_size": (2, 2), "strides": (1, 1), "padding": "valid"},
    )
]

for t in trials:
    print(t.input)
    print(t.result)

with open(os.path.join("truth.pkl"), "wb+") as f:
    pickle.dump(trials, file=f)
