from libraries import *

def _train_KNN(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 23)}

    # Use gridsearch to test all values for n_neighbor
    knn_gs = GridSearchCV(knn, params_knn, cv=5)

    # Fit model to training data
    knn_gs.fit(X_train, y_train)

    # Save best model
    knn_best = knn_gs.best_estimator_


    return knn_best

