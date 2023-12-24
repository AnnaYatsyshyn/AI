import numpy as np
import neurolab as nl

# N E R O
target = [[1, 0, 0, 0, 1,
           1, 1, 0, 0, 1,
           1, 0, 1, 0, 1,
           1, 0, 0, 1, 1,
           1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 0, 0, 0, 1],
          [0, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           0, 1, 1, 1, 0]]

chars = ['N', 'E', 'R', 'O']
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

print("\nTest on defaced N:")
test = np.asfarray([0, 0, 0, 0, 0,
                    1, 1, 0, 0, 1,
                    1, 1, 0, 0, 1,
                    1, 0, 1, 1, 1,
                    0, 0, 0, 1, 1])
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

print("Test on defaced E:")
test_E = np.asfarray([1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0,
                      1, 1, 1, 0, 1,
                      1, 0, 0, 0, 0,
                      1, 1, 0, 1, 1])
test_E[test_E == 0] = -1

out_E = net.sim([test_E])
print((out_E[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))

print("Test on defaced R:")
test_R = np.asfarray([1, 1, 1, 1, 1,
                      1, 0, 0, 1, 1,
                      1, 1, 1, 1, 0,
                      1, 1, 0, 1, 0,
                      1, 0, 0, 0, 1], )
test_R[test_R == 0] = -1

out_R = net.sim([test_R])
print((out_R[0] == target[2]).all(), 'Sim. steps', len(net.layers[0].outs))

print("Test on defaced O:")
test_O = np.asfarray([0, 1, 1, 1, 0,
                      1, 0, 0, 1, 1,
                      1, 0, 1, 1, 1,
                      1, 0, 0, 0, 1,
                      0, 1, 1, 1, 0], )
test_O[test_O == 0] = -1

out_O = net.sim([test_O])
print((out_O[0] == target[3]).all(), 'Sim. steps', len(net.layers[0].outs))
