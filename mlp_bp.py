import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

#Import Data
X_data=pd.read_csv("X_loan_std.csv",delimiter=";")
y_data=pd.read_csv("y_loan.csv",delimiter=";")
X_data=pd.DataFrame(X_data)
y_data=pd.DataFrame(y_data)
X_data=X_data.as_matrix(columns=None)
y_data=y_data.as_matrix(columns=None)

#Prepeocessing
X=preprocessing.StandardScaler().fit(X_data).transform(X_data)

#Split Data into test and training sets
X_train,X_test,y_train,y_test=train_test_split(X,y_data)

#Parameters
learinig_rate = 0.005
regularization = 0.005
epoch=10
batch=1

#Network Parameters
num_examples = len(X_train)  # training set size
n_input = 144   # input layer dim
n_hidden_1=100  # first hidden layer dim
n_hidden_2=40   # second layer dim
n_classes = 7   # output layer dim

# predict the target attribute
def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


#Build a model
def build_model(n_hidden_1,n_hidden_2, epoch):
    # Initialization Weights and biases tensors
    np.random.seed(0)
    W1 = np.random.randn(n_input, n_hidden_1) / np.sqrt(n_input)
    b1 = np.zeros((1, n_hidden_1))
    W2 = np.random.randn(n_hidden_1, n_hidden_2) / np.sqrt(n_hidden_1)
    b2 = np.zeros((1, n_hidden_2))
    W3 = np.random.randn(n_hidden_2, n_classes) / np.sqrt(n_hidden_2)
    b3 = np.zeros((1, n_classes))

    # model dictionary
    model = {}

    # Gradient descent
    for i in range(0, epoch):
        print(i,'epoch')
        # Batch
        total_batch = int(len(X_train) / batch)
        for j in range(total_batch):
            randidx = np.random.randint(int(len(X_train)), size=batch)
            batch_x = X_train[randidx, :]
            batch_y = y_train[randidx]

            # Forward propagation
            z1 = batch_x.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            a2 = np.tanh(z2)
            z3= a2.dot(W3) + b3
            exp_scores = np.exp(z3)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta4 = probs
            delta4[range(batch), batch_y] -= 1
            dW3 = (a2.T).dot(delta4)
            db3 = np.sum(delta4, axis=0, keepdims=True)
            delta3 = delta4.dot(W3.T) * (1 - np.power(a2, 2))
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3,axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T)*(1-np.power(a1,2))
            dW1 =np.dot(batch_x.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms
            dW3 += regularization * W3
            dW2 += regularization * W2
            dW1 += regularization * W1

            # Gradient descent parameter
            W1 += -learinig_rate * dW1
            b1 += -learinig_rate * db1
            W2 += -learinig_rate * dW2
            b2 += -learinig_rate * db2
            W3 += -learinig_rate * dW3
            b3 += -learinig_rate * db3

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
        # Accuaracy
        pred = []
        for x in X_test:
            pred.append(predict(model, x))
        print(confusion_matrix(y_test, pred).trace()/len(y_test)*100,'%acc test')
        pred = []
        for x in X_train:
            pred.append(predict(model, x))
        print(confusion_matrix(y_train, pred).trace() / len(y_train) * 100, '%acc train')
    return model

# Build a model
model = build_model(n_hidden_1,n_hidden_2,epoch)

#Confusion matrix + accuracy
pred=[]
for x in X_test:
    pred.append(predict(model,x))
print (confusion_matrix(y_test,pred))
print (confusion_matrix(y_test,pred).trace())