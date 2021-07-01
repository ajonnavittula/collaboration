import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    filename = sys.argv[1]
    savename = "data/demos/" + filename + ".pkl"
    data = pickle.load(open(savename, "rb"))
    data = np.asarray(data)
    data = data[data[:,0].argsort()]
    data[:,0] = np.degrees(data[:,0])
    plt.scatter(data[:,0], data[:,1])
    plt.xlabel("angle")
    plt.ylabel("confidence")
    plt.title("Confidence in correct goal vs. starting angle")
    plt.show()

if __name__ == '__main__':
    main()