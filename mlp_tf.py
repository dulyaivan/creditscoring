from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

X_data=pd.read_csv("X_loan_std.csv",delimiter=";")
y_data=pd.read_csv("y_loan.csv",delimiter=";")
X_data=pd.DataFrame(X_data)
y_data=pd.DataFrame(y_data)
X_data=X_data.as_matrix(columns=None)
y_data=y_data.as_matrix(columns=None)


#Split data into test and training sets
X_train,X_test,y_train,y_test=train_test_split(X_data,y_data,random_state=0)

#Transition to boolean space for the target
df_y=pd.DataFrame(y_train,columns=['y'])
uniq_list=df_y["y"].unique()
for j in uniq_list:
    d = {j: 1}
    df_y[j] = df_y["y"].map(d).replace(to_replace=['NaN'], value=0)
del df_y["y"]
y_train=df_y.as_matrix()


# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 20
display_step = 1

# Network Parameters
n_hidden_1 = 100
n_hidden_2 = 100
n_input = 144
n_classes = 7

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            randidx = np.random.randint(int(len(X_train)), size=batch_size)
            batch_x = X_train[randidx, :]
            batch_y = y_train[randidx, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

# Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
# Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: X_test, Y: y_test}))
    prediction = tf.argmax(logits, 1)
    best = sess.run([prediction], {X: X_test, Y: y_test})

print(best)
#best_num=bool_to_num(best)
print(confusion_matrix(y_test,best[0]))
print(confusion_matrix(y_test,best[0]).trace())
print("acc%:",confusion_matrix(y_test,best[0]).trace()/len(y_test)*100)