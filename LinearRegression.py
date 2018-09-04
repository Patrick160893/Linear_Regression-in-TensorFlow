import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

projectDir = os.path.dirname(os.path.realpath(__file__))

# TRAFFIC
# df = pd.read_excel(projectDir + '/slr05.xls')
# df.dropna(inplace=True)
# learningRate = 0.001
# numEpochs = 5000

# BURGLARIESí
df = pd.read_excel(projectDir + '/Offences.xls')
df = df.loc[:, ['Burglary', 'Murder and\nnonnegligent\nmanslaughter']]
df.columns = ['Burlaries', 'Manslaughter']
df.dropna(inplace=True)
learningRate = 0.0000001
numEpochs = 100

x = tf.placeholder(tf.float32, shape=[None, ])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.zeros(shape=[1]))
b = tf.Variable(tf.zeros(shape=[1]))

y_pred = tf.add(tf.multiply(x, w), b)

loss = tf.reduce_mean(tf.square(tf.subtract(y, y_pred)))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    input, output = df.iloc[:, 0], df.iloc[:, 1]
    for epoch in range(numEpochs):
        _, sampleLoss, pred = session.run([optimiser, loss, y_pred], {x: input, y: output})
        print("EPOCH:", epoch + 1)
        print("LOSS: ", sampleLoss)
        print("")

    y_preds = session.run(y_pred, {x: input, y: output})

for i in range(df.shape[0]):
    x, y = df.iloc[i, 0], df.iloc[i, 1]
    plt.scatter(x, y, c='b')

plt.plot(input, y_preds, c='r')

plt.show()
