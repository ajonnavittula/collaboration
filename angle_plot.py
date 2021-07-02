import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    filenames = ["10", "20", "30", "45", "60", "90"]
    fig, axs = plt.subplots(2,3)
    axs = axs.reshape(-1)
    # filename = sys.argv[1]
    for i,filename in enumerate(filenames):
        ax = axs[i]
        filename = filename + "deg"
        savename = "data/demos/" + filename + ".pkl"
        data = pickle.load(open(savename, "rb"))
        data = np.asarray(data)
        data = data[data[:,0].argsort()]
        data[:,0] = np.degrees(data[:,0])
        # plt.scatter(data[:,0], data[:,1])
        ax.hist(data[:,1], 10)
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        ax.set_title(filename)
    plt.suptitle("Histograms of confidence over 250 trials for intial angle + noise")
    plt.show()

if __name__ == '__main__':
    main()