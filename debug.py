import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import tensorflow as tf

import gpflow
from gpflow.utilities import read_values, multiple_assign, set_trainable
from gpflow.inducing_variables import InducingPoints
import gpmaniflow
from gpmaniflow.models.SVGP import SVGP

#pX = np.linspace(-5.0, 5.0, 200)[:, None]
pX = np.reshape(np.random.uniform(-3, 3, 200*2), [200, 2])
X = np.random.uniform(-3, 3, 100*2)[:, None]
X = np.reshape(X, [100, 2])
Y = 2 * X[:,0] + 1 * X[:,1] + np.random.randn(100) * 0.1
Y = np.reshape(Y, [100,1])
Z = np.random.uniform(-3, 3, 50*2)[:, None]
Z = np.reshape(Z, [50, 2])

train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
train_dataset = train_dataset.shuffle(1024).batch(len(X))

#plt.plot(X,Y ,"x")

kernel = gpflow.kernels.SquaredExponential()

model = SVGP(kernel, likelihood = gpflow.likelihoods.Gaussian(), inducing_variable=InducingPoints(Z.copy()))

train_iter = iter(train_dataset.repeat())
training_loss = model.training_loss_closure(train_iter, compile=True)
optimizer = tf.keras.optimizers.Adam(0.01)
@tf.function
def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)
elbo_hist = []
for step in range(2000):
    optimization_step()
    if step % 50 == 0:
        minibatch_elbo = -training_loss().numpy()
        print('Step: %s, Mini batch elbo: %s' % (step, minibatch_elbo))
        elbo_hist.append(minibatch_elbo)

ELBO = model.elbo((X,Y))
dmu, dvar = model.predict_df(pX)