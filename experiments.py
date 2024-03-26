import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(path):
    data = np.loadtxt(path)
    # Set the seaborn style
    sns.set(style="whitegrid")

    # Create a colormap
    cmap = sns.color_palette("viridis_r", as_cmap=True)

    plt.figure(figsize=(15, 5))

    # Plot histogram
    plt.subplot(1, 4, 1)
    plt.hist(data, bins=30, color=cmap(0.5), edgecolor="black")
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    # Plot distribution plot
    plt.subplot(1, 4, 2)
    sns.distplot(data, hist=False, color="red", kde_kws={"shade": True})
    plt.title("Distribution Plot")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()
