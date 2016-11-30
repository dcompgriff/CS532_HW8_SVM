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

    #Build LS classifier, and find w.
    w = np.dot(np.linalg.pinv(reducedData[:, :-1]), reducedData[:, -1])

    #Generate decision boundary by passing in x from 0 to 8 at .1 spacing np.linspace(0, 8, 80)
    #Classify as np.dot(x, w) > 0 = 1, or np.dot(x, w) < 0 = -1
    xList = np.linspace(0, 8, 80)
    yList = []
    for x in xList:
        #Given x, solve for y.
        yvalue = (1.0/w[1])*(-w[2]-(x*w[0]))
        yList.append(yvalue)

    #Build plot.
    plt.ylabel('feature 4 (petal width)')
    plt.xlabel('feature 3 (petal length)')
    plt.ylim((0, 3.5))
    plt.xlim((2, 8))
    #Plot data points.
    plt.scatter(reducedData[:50, 0], reducedData[:50, 1], color ='r')
    plt.scatter(reducedData[50:, 0], reducedData[50:, 1], color='b')
    #Plot decision boundary.
    plt.plot(xList, yList, color='black')

    #Plot legend.
    red_patch = mpatches.Patch(color='red', label='virginica')
    blue_patch = mpatches.Patch(color='blue', label='versicolor')
    plt.legend(handles=[red_patch, blue_patch], loc=2)
    plt.show()


    print(data)











if __name__ == '__main__':
    main()