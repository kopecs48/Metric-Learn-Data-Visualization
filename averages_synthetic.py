from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(*load_wine(return_X_y=True), test_size=.70)
scores = [0.97967,0.97567,0.98967]
methods = ['LMNN', 'NCA', 'LFDA', 'KNN']
knn_scores = []
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    knn_scores.append(knn.score(X_test,y_test))

total = 0
for score in knn_scores:
    total += score
knn_average = total / 10
scores.append(knn_average)
pprint(scores)

fig = plt.figure()
fig.set_size_inches(20, 12, forward=True)
fig.set_dpi(100)
st = fig.suptitle("Method Comparison", fontsize="x-large")
ax1 = fig.add_subplot(311)
ax1.bar(methods, scores)
ax1.set_title("Accuracy Score")
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Method")
ax1.set_ylim(.75,1)

fig.tight_layout()

# shift subplots down:
st.set_y(1)
fig.subplots_adjust(top=.95)

fig.savefig("averages_synthetic.png")