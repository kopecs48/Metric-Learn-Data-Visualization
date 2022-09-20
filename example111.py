from metric_learn import LMNN
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


X, y = make_classification(n_samples=1000, n_classes=3, n_clusters_per_class=2,
                           n_informative=3, class_sep=4., n_features=5,
                           n_redundant=0, shuffle=True,
                           scale=[1, 1, 20, 20, 20])
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=.70)
lmnn_knn = Pipeline(steps=[('lmnn', LMNN()), ('knn', KNeighborsClassifier())])
parameters = {'lmnn__k':[1,2,3,4,5], 'knn__n_neighbors':[1,2,3,4,5,6,7,8]}
grid_lmnn_knn = GridSearchCV(lmnn_knn, parameters, scoring='accuracy', cv=5, n_jobs=-1, verbose=True, return_train_score=True)
grid_lmnn_knn.fit(X_train, y_train)
pred = grid_lmnn_knn.predict(X_test)
grid_lmnn_knn.score(X_test, y_test)
