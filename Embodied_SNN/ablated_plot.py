import matplotlib.pyplot as plt
import numpy

X = [833.8, 1920.0, 1633.2, 736.6]

X_ablated = [966.2, 462.8, 1402.8, 1539.8]

plt.scatter([1] * len(X), X, color='blue')
plt.scatter([1], [numpy.mean(X)], color='red')
plt.scatter([2] * len(X_ablated), X_ablated, color='blue')
plt.scatter([2], [numpy.mean(X_ablated)], color='red')

for score, score_ablated in zip(X, X_ablated):
    plt.plot([1, 2], [score, score_ablated], color='blue')
plt.plot([1, 2], [numpy.mean(X), numpy.mean(X_ablated)], color='red', label="Mean")

plt.xticks([1, 2], ['Recurrent', 'Ablated'])
plt.xlim(0.5, 2.5)
plt.ylabel('Score')

# Hide all spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.legend()
plt.show()
