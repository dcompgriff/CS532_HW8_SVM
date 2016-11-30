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
    reducedData = np.vstack((data[100:,2:], data[50:100,2:]))
    #Build plot.
    plt.ylabel('feature 4 (petal width)')
    plt.xlabel('feature 3 (petal length)')
    plt.ylim((0, 3.5))
    plt.xlim((2, 8))
    plt.scatter(reducedData[:50, 0], reducedData[:50, 1], color ='r')
    plt.scatter(reducedData[50:, 0], reducedData[50:, 1], color='b')
    red_patch = mpatches.Patch(color='red', label='virginica')
    blue_patch = mpatches.Patch(color='blue', label='versicolor')
    plt.legend(handles=[red_patch, blue_patch], loc=2)
    plt.show()


    print(data)











if __name__ == '__main__':
    main()