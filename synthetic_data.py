from pprint import pprint
import numpy as np
import metric_learn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from metric_learn import LMNN, LFDA, NCA
import metric_learn
np.random.seed(42)

# X, y = make_classification(n_samples=1000, n_classes=3, n_clusters_per_class=2,
#                            n_informative=3, class_sep=4., n_features=5,
#                            n_redundant=0, shuffle=True,
#                            scale=[1, 1, 20, 20, 20])
# # X_train, X_test, y_train, y_test = train_test_split(*load_wine(return_X_y=True))
# X_train, X_test, y_train, y_test = train_test_split(X,y)
# lmnn_knn = Pipeline(steps=[('lmnn', LFDA()), ('knn', KNeighborsClassifier())])
# parameters = {'lfda__k':[1,2], 'lfda__n_components':[1,2], 'knn__n_neighbors':[1,2]}
# grid_lmnn_knn = GridSearchCV(lmnn_knn, parameters, cv=3, n_jobs=-1, verbose=True)
# grid_lmnn_knn.fit(X_train, y_train)
# print('Metric')
# print(grid_lmnn_knn.score(X_test, y_test))



# for i in range(1,11):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     print('knn: ' , i)
#     print(knn.score(X_test, y_test))

scores = []
best_params = []
cv_results = []
best_scores = []
for i in range(1,2):
    X, y = make_classification(n_samples=1000, n_classes=3, n_clusters_per_class=2,
                           n_informative=3, class_sep=4., n_features=5,
                           n_redundant=0, shuffle=True,
                           scale=[1, 1, 20, 20, 20])
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=.70)
    lmnn_knn = Pipeline(steps=[('lmnn', LMNN()), ('knn', KNeighborsClassifier())])
    parameters = {'lmnn__k':[1,2,3], 'knn__n_neighbors':[1,2,3,4,5]}
    grid_lmnn_knn = GridSearchCV(lmnn_knn, parameters, scoring='accuracy', cv=5, n_jobs=-1, verbose=True, return_train_score=True)
    grid_lmnn_knn.fit(X_train, y_train)
    
    best_params.append(grid_lmnn_knn.best_params_)
    scores.append(grid_lmnn_knn.score(X_test, y_test))
    cv_results.append(grid_lmnn_knn.cv_results_)
    best_scores.append(grid_lmnn_knn.best_estimator_)

pprint(scores)
print(best_params)
print(best_scores)
# pprint(cv_results)
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    print('knn: ' , i)
    print(knn.score(X_test, y_test))

