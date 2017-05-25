import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import time
import pickle

current = time.time()


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # getting the number of features
    n_feature = mat.get("train1").shape[1]

    # getting the size of data set
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    # print("Size of Validation set: ", validation_data.shape)
    # print("Size of Validation Label: ", validation_label.shape)
    # print("Size of Training Data: ", train_data.shape)
    # print("Size of Training Label: ", train_label.shape)
    # print("Size of Test Data: ", test_data.shape)
    # print("Size of Test Label: ", test_label.shape)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Adding bias term
    train_data = np.insert(train_data, 0, 1, axis=1)

    # Computing hypothesis
    Xw = np.dot(train_data, initialWeights)
    hypothesis = np.array(sigmoid(Xw)).reshape((n_data, 1))

    # Computing error
    error_comp1 = np.dot(np.transpose(labeli), np.log(hypothesis))
    error_comp2 = np.dot(np.transpose(1 - labeli), np.log(1 - hypothesis))
    error = -(error_comp1 + error_comp2) / n_data

    # Computing gradient
    diff = np.subtract(hypothesis, labeli)
    error_grad = (np.dot(np.transpose(train_data), diff)) / n_data

    # print("error grad: ", error_grad.shape)

    return np.asscalar(error), error_grad.flatten()


def blrPredict(W, data):
    """
     blrPredict predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Adding bias term
    data = np.insert(data, 0, 1, axis=1)

    # Computing the probability Matrix
    Xw = np.dot(data, W)
    probab_matrix = sigmoid(Xw)

    # Taking the class corresponding to the largest probability
    label = np.argmax(probab_matrix, axis=1).reshape((data.shape[0], 1))

    return label


def mlrObjFunction(initWeight, *args):
    train_data, labeli = args
    initWeight = initWeight.reshape((train_data.shape[1] + 1, 10))

    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x K
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x K  - 1 of k encoding scheme

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Adding bias term
    train_data = np.insert(train_data, 0, 1, axis=1)

    theta_num = np.exp(np.dot(train_data, initWeight))  # N x K
    theta_deno = np.sum(theta_num, axis=1).reshape((theta_num.shape[0], 1))  # N x 1
    theta = theta_num / theta_deno  # N x K

    # Computing error
    error = -np.sum(np.sum(labeli * np.log(theta), axis=1))

    # Computing error gradient: X^T . (theta - Y)
    error_grad = np.dot(np.transpose(train_data), np.subtract(theta, labeli))

    return error, error_grad.flatten()


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data = np.insert(data, 0, 1, axis=1)

    # Computing the probability Matrix
    theta_num = np.exp(np.dot(data, W))  # N x K
    theta_deno = np.sum(theta_num, axis=1).reshape((theta_num.shape[0], 1))  # N x 1
    probab_matrix = theta_num / theta_deno  # N x K

    # Taking the class corresponding to the largest probability
    label = np.argmax(probab_matrix, axis=1).reshape((data.shape[0], 1))

    return label


# Method to calculate accuracy
def predictionAccuracy(predicted_1, predicted_2, predicted_3):
    # Find the accuracy on Training Dataset
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_1 == train_label).astype(float))) + '%')

    # Find the accuracy on Validation Dataset
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_2 == validation_label).astype(float))) + '%')

    # Find the accuracy on Testing Dataset
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_3 == test_label).astype(float))) + '%')


# Support Vector Machines
def _svm(clf):
    # Learn model with given params on training data
    clf.fit(train_data, train_label.ravel())

    # Prediction on training data
    predicted_1 = clf.predict(train_data)
    predicted_1 = predicted_1.reshape((predicted_1.shape[0], 1))

    # Prediction on validation data
    predicted_2 = clf.predict(validation_data)
    predicted_2 = predicted_2.reshape((predicted_2.shape[0], 1))

    # Prediction on test data
    predicted_3 = clf.predict(test_data)
    predicted_3 = predicted_3.reshape((predicted_3.shape[0], 1))

    # Calculating accuracy using labels
    predictionAccuracy(predicted_1, predicted_2, predicted_3)


def calc_time_taken(current, previous, str):
    # Calculating time taken
    diff = current - previous
    m, s = divmod(diff, 60)
    h, m = divmod(m, 60)
    print(str, " ran for %d:%02d:%02d" % (h, m, s), end="\n")


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

print("train data:", train_data.shape)
print("train label: ", train_label.shape)

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

print('\n\n--------------BLR-------------------\n\n')

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}

for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label_train = blrPredict(W, train_data)
# Find the accuracy on Validation Dataset
predicted_label_valid = blrPredict(W, validation_data)
# Find the accuracy on Testing Dataset
predicted_label_test = blrPredict(W, test_data)
# calling prediction function
predictionAccuracy(predicted_label_train, predicted_label_valid, predicted_label_test)

# Writing the learned weight to a pickle file
f1 = open('params.pickle', 'wb')
pickle.dump(W, f1)
f1.close()

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
# ##################
# # YOUR CODE HERE #
# ##################

print("Started Running SVM")

# Requirement1 - use linear kernel with default parameters
req1 = svm.SVC(kernel='linear')
_svm(req1)
calc_time_taken(time.time(), current, " Requirement 1: Linear Kernel")
current = time.time()

# Requirement2 -
req2 = svm.SVC(kernel='rbf',gamma=1.0)
_svm(req2)
calc_time_taken(time.time(), current, " Requirement 2: RBF Kernel, Gamma-1.0")
current = time.time()

# Requirement3 -
req3 = svm.SVC(kernel='rbf')
_svm(req3)
calc_time_taken(time.time(), current, " Requirement 3: RBF Kernel")
current = time.time()


req4 = svm.SVC(kernel='rbf',C=1.0)
_svm(req4)
calc_time_taken(time.time(), current, "Requirement 4 with C: 1")
current = time.time()

# Requirement4 -
for index in range(0, 110, 10):
    if index == 0:
        continue
    req4 = svm.SVC(kernel='rbf',C=float(index))
    _svm(req4)
    calc_time_taken(time.time(), current, "Requirement 4 with C: " + str(index))
    current = time.time()


print("Ended running SVM")

# """
# Script for Extra Credit Part
# """
print('\n\n--------------MLR-------------------\n\n')
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
# mlrObjFunction(initialWeights_b, train_data, Y)

nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_train = mlrPredict(W_b, train_data)
# Find the accuracy on Validation Dataset
predicted_label_valid = mlrPredict(W_b, validation_data)
# Find the accuracy on Testing Dataset
predicted_label_test = mlrPredict(W_b, test_data)
# calling prediction function
predictionAccuracy(predicted_label_train, predicted_label_valid, predicted_label_test)


f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
