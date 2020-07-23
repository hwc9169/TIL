import numpy as np

timesteps = 10
input_size = 4
hidden_size = 8

inputs = np.random.random((timesteps,input_size))

hidden_state_t = np.zeros((hidden_size,))

Wx = np.random.random((hidden_size,input_size))
Wh = np.random.random((hidden_size,hidden_size))
b = np.random.random((hidden_size,))

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b)

    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states))

    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states)
print(total_hidden_states)