import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math






def main():
    #q1()
    q2()


def q2():
    data = np.loadtxt('./circledata.csv', delimiter=',')
    # data = np.loadtxt('./fisher.csv', delimiter=',')
    # data = data[:, 2:]
    # labels = np.vstack((np.ones((50, 1)), -1*np.ones((100, 1))))
    # data = np.hstack((data, labels))
    data = np.hstack((np.ones((data.shape[0], 1)), data))

    '''
    PART A
    '''
    #Standard LS regularization solution.
    lambdaParam = 10**-5
    X = data[:, :-1]
    y = data[:, -1]
    w1 = np.linalg.inv(X.T.dot(X) + lambdaParam*np.eye(X.T.dot(X).shape[0], X.T.dot(X).shape[1])).dot(X.T).dot(y)
    #Dual LS regularization solution.
    K = np.dot(X, X.T)
    alpha = np.linalg.inv(K + lambdaParam*np.eye(K.shape[0], K.shape[1])).dot(y)
    w2 = X.T.dot(alpha)
    #Print w values to show they are equal.
    print('w1:')
    print(w1)
    print('w2:')
    print(w2)
    #Perform classification for original LS case and record the accuracy.
    primalPredicted = []
    for x in X:
        primalPredicted.append(np.sign(np.dot(x, w1)))
    originalLSAccuracy = calculateAccuracy(y, primalPredicted)

    #Perform classification for dual LS case and compare results to primal classification results.
    dualPredicted = []
    for x in X:
        sumTerm = 0
        for i in range(0, X.shape[0]):
            sumTerm += np.dot(x, X[i])*alpha[i]
        dualPredicted.append(np.sign(sumTerm))
    equal = np.allclose(np.array(dualPredicted), np.array(primalPredicted))
    print('Primal and Dual Classifications equal?: ' + str(equal))

    '''
    PART B, Gaussian kernel
    '''
    kij = lambda xi, xj: math.exp((-.5)*(np.linalg.norm(xi - xj)**2))
    #Build Gaussian kernel matrix for each pair of entries in matrix X.
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[0]):
            K[i, j] = kij(X[i], X[j])

    alpha = np.linalg.inv(K + lambdaParam * np.eye(K.shape[0], K.shape[1])).dot(y)
    gaussianKernelPredicted = []
    for x in X:
        sumTerm = 0
        for i in range(0, X.shape[0]):
            sumTerm += kij(x, X[i]) * alpha[i]
        gaussianKernelPredicted.append(np.sign(sumTerm))
    gaussianKernelAccuracy = calculateAccuracy(y, gaussianKernelPredicted)
    print('Linear Kernel Accuracy: %.10f' % originalLSAccuracy)
    print('Gaussian Kernel Accuracy: %.10f' % gaussianKernelAccuracy)

    '''
    PART C, quadratic kernel
    '''
    kij = lambda xi, xj: (np.dot(xi, xj) + 1)**2
    #Build Gaussian kernel matrix for each pair of entries in matrix X.
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[0]):
            K[i, j] = kij(X[i], X[j])

    alpha = np.linalg.inv(K + (lambdaParam * np.eye(K.shape[0], K.shape[1]) )).dot(y)
    quadraticKernelPredicted = []
    for x in X:
        sumTerm = 0
        for i in range(0, X.shape[0]):
            sumTerm += ((np.dot(x, X[i]) + 1)**2)*alpha[i]
            #sumTerm += kij(x, X[i]) * alpha[i]
        quadraticKernelPredicted.append(np.sign(sumTerm))
    quadraticKernelAccuracy = calculateAccuracy(y, quadraticKernelPredicted)
    print('Quadratic Kernel Accuracy: %.10f' % quadraticKernelAccuracy)

    temp = np.array(gaussianKernelPredicted)
    plt.scatter(X[temp == 1, 1], X[temp == 1, 2], color='b')
    plt.scatter(X[temp == -1, 1], X[temp == -1, 2], color='r')
    plt.title('Gaussian kernel classification.')
    plt.show()

    temp = np.array(quadraticKernelPredicted)
    plt.scatter(X[temp == 1, 1], X[temp == 1, 2], color='b')
    plt.scatter(X[temp == -1, 1], X[temp == -1, 2], color='r')
    plt.title('Quadratic kernel classification.')
    plt.show()



    #C = 1/lambda for scikit-learn SVM

    print('Done!')




def q1():
    #Part A make the iris data set plot.
    data = np.loadtxt('./fisher.csv', delimiter=',')
    # labels = np.vstack((-1*np.ones((50, 1)), np.zeros((50, 1))))
    # labels = np.vstack((labels, np.ones((50, 1))))
    #Add labels to the data set.
    #data = np.hstack((data, labels))

    #Condense data set into +1 labels, -1 labels, and 3rd, 4th features.
    labels = np.vstack((-1 * np.ones((50, 1)), np.ones((50, 1))))
    reducedData = np.vstack((data[100:,2:], data[50:100,2:]))
    #row = (x1, x2, 1, label)
    reducedData = np.hstack((reducedData, np.ones((100, 1)), labels))

    q1_pa(reducedData)
    q1_pb(reducedData)
    q1_pc(reducedData)

'''
wt convergence properties.
'''
def q1_pc(reducedData):
    #Get w values for initial parameters.
    w, wconvergence = trainSVM(reducedData, returnConvergence=True)
    #Plot w1, w2, and w3
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[0], wconvergence)), color='r')
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[1], wconvergence)), color='g')
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[2], wconvergence)), color='b')
    plt.xlabel('Iteration')
    plt.ylabel('w value')
    plt.title('W convergence gamma=0.003')
    plt.show()

    #Get w values for gamma = 0.01
    w, wconvergence = trainSVM(reducedData, gammaParam=0.01, returnConvergence=True)
    plt.clf()
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[0], wconvergence)), color='r')
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[1], wconvergence)), color='g')
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[2], wconvergence)), color='b')
    plt.xlabel('Iteration')
    plt.ylabel('w value')
    plt.title('W convergence gamma=0.01')
    plt.show()

    # Get w values for gamma = 0.0001
    w, wconvergence = trainSVM(reducedData, gammaParam=0.0001, returnConvergence=True)
    plt.clf()
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[0], wconvergence)), color='r')
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[1], wconvergence)), color='g')
    plt.scatter(list(range(0, len(wconvergence))), list(map(lambda row: row[2], wconvergence)), color='b')
    plt.xlabel('Iteration')
    plt.ylabel('w value')
    plt.title('W convergence gamma=0.0001')
    plt.show()


'''
Regularized SVM training with gradient descent.
'''
def q1_pb(reducedData):
    #Get w for data.
    w = trainSVM(reducedData)

    #Plot SVM line.
    xList = np.linspace(0, 8, 80)
    yList = []
    for x in xList:
        # Given x, solve for y.
        yvalue = (1.0 / w[1]) * (-w[2] - (x * w[0]))
        yList.append(yvalue)

    # Build plot.
    plt.ylabel('feature 4 (petal width)')
    plt.xlabel('feature 3 (petal length)')
    plt.ylim((0, 3.5))
    plt.xlim((2, 8))
    # Plot data points.
    plt.scatter(reducedData[:50, 0], reducedData[:50, 1], color='r')
    plt.scatter(reducedData[50:, 0], reducedData[50:, 1], color='b')
    # Plot decision boundary.
    plt.plot(xList, yList, color='black')

    # Plot legend.
    red_patch = mpatches.Patch(color='red', label='virginica')
    blue_patch = mpatches.Patch(color='blue', label='versicolor')
    plt.legend(handles=[red_patch, blue_patch], loc=2)
    plt.show()

def trainSVM(reducedData, lambdaParam = 0.1, gammaParam = 0.003, iterations=20000, returnConvergence=False):
    # Initialize w0 to 0 vector.
    w = np.zeros((3, 1))
    wconvgList = []

    for t in range(0, iterations):
        if t % 100 == 0:
            print('Iteration %d' % t)
        grad = 0
        errorSumTerm = 0
        for i in range(0, reducedData.shape[0]):
            # Calculate summed error term over training examples.
            errorSumTerm += -reducedData[i, -1] * reducedData[i, :-1].reshape((reducedData.shape[1] - 1, 1)) * \
                            (.5 * (1 + np.sign(1 - (reducedData[i, -1] * np.dot(reducedData[i, :-1], w)))))[0]
        # Calculate regularization term.
        wErrorTerm = 2 * lambdaParam * w
        wErrorTerm[-1, 0] = 0
        # Calculate gradient.
        grad = errorSumTerm + wErrorTerm
        # Update w term.
        w = w - (gammaParam * grad)
        if returnConvergence:
            wconvgList.append(w)

    if returnConvergence:
        return w, wconvgList
    else:
        return w

def q1_pa(reducedData):
    # Build LS classifier, and find w.
    w = np.dot(np.linalg.pinv(reducedData[:, :-1]), reducedData[:, -1])

    # Generate decision boundary by passing in x from 0 to 8 at .1 spacing np.linspace(0, 8, 80)
    # Classify as np.dot(x, w) > 0 = 1, or np.dot(x, w) < 0 = -1
    xList = np.linspace(0, 8, 80)
    yList = []
    for x in xList:
        # Given x, solve for y.
        yvalue = (1.0 / w[1]) * (-w[2] - (x * w[0]))
        yList.append(yvalue)

    # Build plot.
    plt.ylabel('feature 4 (petal width)')
    plt.xlabel('feature 3 (petal length)')
    plt.ylim((0, 3.5))
    plt.xlim((2, 8))
    # Plot data points.
    plt.scatter(reducedData[:50, 0], reducedData[:50, 1], color='r')
    plt.scatter(reducedData[50:, 0], reducedData[50:, 1], color='b')
    # Plot decision boundary.
    plt.plot(xList, yList, color='black')

    # Plot legend.
    red_patch = mpatches.Patch(color='red', label='virginica')
    blue_patch = mpatches.Patch(color='blue', label='versicolor')
    plt.legend(handles=[red_patch, blue_patch], loc=2)
    plt.show()

'''
Calculate straight number of times that two class labels agreed.
'''
def calculateAccuracy(yactual, ypredicted):
	metrics = {}
	metrics["accuracy"] = 0

	for i in range(0, len(yactual)):
		if ypredicted[i] == yactual[i]:
			metrics["accuracy"] += 1

	metrics["accuracy"] = metrics["accuracy"] / float(len(yactual))
	return metrics["accuracy"]










if __name__ == '__main__':
    main()