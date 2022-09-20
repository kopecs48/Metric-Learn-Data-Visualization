import collections
from pprint import pprint
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from metric_learn import LMNN, LFDA, NCA
import metric_learn

import numpy as np
import matplotlib.pyplot as plt
scores = []
best_params = []
f1 = []
cv_results = []
best_scores = []
for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(*load_wine(return_X_y=True))
    lmnn_knn = Pipeline(steps=[('lmnn', LMNN()), ('knn', KNeighborsClassifier())])
    parameters = {'lmnn__k':[1,2,3,4,5,6,7,8,9,10,11,12], 'knn__n_neighbors':[1,2,3,4,5,6,7,8]}
    grid_lmnn_knn = GridSearchCV(lmnn_knn, parameters, scoring='accuracy', cv=3, n_jobs=-1, verbose=True, return_train_score=True)
    grid_lmnn_knn.fit(X_train, y_train)

    pred = grid_lmnn_knn.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=pred)
    scores.append(accuracy)
    f1_val = f1_score(y_test, pred, average='weighted')
    f1.append(f1_val)

    best_params.append(grid_lmnn_knn.best_params_)
    # scores.append(grid_lmnn_knn.score(X_test, y_test))
    cv_results.append(grid_lmnn_knn.cv_results_)
    best_scores.append(grid_lmnn_knn.best_estimator_)
    pprint(confusion_matrix(y_test, pred))
# pprint('Metric:')
pprint(scores)
pprint(f1)

# pprint(best_params)
# pprint(best_scores)
#,6,7,8,9,10,11,12
#,3,4,5,6,7,8

f1_frequency = dict(collections.Counter(f1))
f1_frequency = sorted(f1_frequency.items())
f1_percents = [i[0] for i in f1_frequency]
f1_values = [i[1] for i in f1_frequency]
f1_percents = [str(i) for i in f1_percents]

score_frequency = dict(collections.Counter(scores))
score_frequency = sorted(score_frequency.items())

score_percents = [i[0] for i in score_frequency]
score_values = [i[1] for i in score_frequency]
score_percents = [str(i) for i in score_percents]

params_frequency = [str(i) for i in best_params]
params_frequency = dict(collections.Counter(params_frequency))
param_names = params_frequency.keys()
param_values = params_frequency.values()
pprint(score_frequency)
# plt.title("LMNN into KNN")
# plt.subplot(1,2,1)
# plt.bar(score_percents, score_values)
# plt.ylabel("Frequency")
# plt.xlabel("Accuracy Score")
# plt.subplot(1,2,2)
# plt.ylabel("Frequency")
# plt.xlabel("Hyper Parameters")
# plt.bar(param_names, param_values)
# plt.show()
fig = plt.figure()
fig.set_size_inches(18.5, 10.5, forward=True)
fig.set_dpi(100)
st = fig.suptitle("LMNN to KNN", fontsize="x-large")
ax1 = fig.add_subplot(311)
ax1.bar(score_percents, score_values)
ax1.set_title("Accuracy Frequency")
ax1.set_ylabel("Frequency")
ax1.set_xlabel("Accuracy Score")

ax1.set_xticklabels(score_percents, rotation = 45, fontsize=12)


ax2 = fig.add_subplot(312)
ax2.bar(param_names, param_values)
ax2.set_title("Parameters Frequency")
ax2.set_ylabel("Frequency")
ax2.set_xlabel("Hyperparameters")
ax2.set_xticklabels(param_names, rotation = 45, fontsize=12)

ax3 = fig.add_subplot(313)
ax3.bar(f1_percents, f1_values)
ax3.set_title("F1 Frequency")
ax3.set_ylabel("Frequency")
ax3.set_xlabel("F1 Score")
ax3.set_xticklabels(f1_percents, rotation = 45, fontsize=12)

fig.tight_layout()

# shift subplots down:
st.set_y(1)
fig.subplots_adjust(top=0.95)

fig.savefig("test.png")



for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    print('knn: ' , i)
    print(knn.score(X_test, y_test))