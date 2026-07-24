import matplotlib.pyplot as plt

marks = [60, 70, 75, 80, 85, 90, 95, 75, 80, 85]

plt.hist(marks, bins=5)
plt.title("Histogram")
plt.show()