# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
# print(char_vocab)

input_size = vocab_size
hidden_size = 5
output_size = 5
learning_rate = 0.1

char_to_index = { v : i for i, v in enumerate(char_vocab)}
# print(char_to_index)

index_to_char = {i : v for i, v in enumerate(char_vocab)}
# print(index_to_char)

x_data = [char_to_index[i] for i in list(input_str)]
y_data = [char_to_index[i] for i in list(label_str)]
# print(x_data)
# print(y_data)

x_data.

