import matplotlib.pyplot as plt


def scatter_plot(pre, label):
    plt.scatter(range(len(label)), label, c="r")
    plt.scatter(range(len(pre)), pre, c="g")
    plt.ylabel("Prediction")
    plt.xlabel("serial number")
    plt.title("error graph")
    plt.show()