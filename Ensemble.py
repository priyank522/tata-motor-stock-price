from libraries import *

def _ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test):
    # Create a dictionary of our models
    estimators = [('knn', knn_model), ('rf', rf_model)]

    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')

    # fit model to training data
    ensemble.fit(X_train, y_train)

    return ensemble