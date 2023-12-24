import numpy as np
import neurolab as nl

# Я А С
target = [[0, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           0, 1, 1, 1, 1,
           0, 1, 0, 0, 1,
           1, 0, 0, 0, 1],
          [0, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1],
          [0, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 0, 0, 0, 0,
           1, 0, 0, 0, 0,
           0, 1, 1, 1, 1]]

chars = ['Я', 'А', 'С']
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

print("\nTest on defaced Я:")
test_YA = np.asfarray([0, 1, 1, 1, 1,
                      1, 0, 0, 0, 1,
                      0, 1, 0, 1, 1,
                      0, 1, 0, 0, 1,
                      1, 0, 1, 0, 1])
test_YA[test_YA == 0] = -1
out_YA = net.sim([test_YA])
print((out_YA[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

print("Test on defaced А:")
test_A = np.asfarray([0, 1, 1, 1, 0,
                      1, 0, 0, 0, 1,
                      1, 0, 1, 1, 1,
                      1, 0, 0, 1, 1,
                      1, 0, 0, 0, 1])
test_A[test_A == 0] = -1

out_A = net.sim([test_A])
print((out_A[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))

print("\nTest on defaced С:")
test_C = np.asfarray([0, 1, 1, 1, 1,
                      0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0,
                      1, 0, 0, 0, 0,
                      0, 1, 0, 1, 1])
test_C[test_C == 0] = -1
out_C = net.sim([test_C])
print((out_C[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))