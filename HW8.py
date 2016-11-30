import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches






def main():
    q1()


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

    #q1_pa(reducedData)
    #q1_pb(reducedData)
    #q1_pc(reducedData)

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












if __name__ == '__main__':
    main()