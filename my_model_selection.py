from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer

from .hyper_parameters_utils import *

def select_model(X_train, y_train, X_test, y_test, mode):
    if mode == 0:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train)
        X_train_imputed = imputer.transform(X_train)
        naive_bayes = GaussianNB()
        k_neighbors = KNeighborsClassifier()
        logistic_regression = LogisticRegression()
        mlp = MLPClassifier(hidden_layer_sizes=(1,))
        svc = SVC()
        decision_tree = DecisionTreeClassifier()
        random_forest = RandomForestClassifier()
        model_naive_bayes = naive_bayes.fit(X_train_imputed, y_train)
        model_k_neighbors = k_neighbors.fit(X_train_imputed, y_train)
        model_logistic_regression = logistic_regression.fit(X_train_imputed, y_train)
        model_mlp = mlp.fit(X_train_imputed, y_train)
        model_svc = svc.fit(X_train_imputed, y_train)
        model_decision_tree = decision_tree.fit(X_train_imputed, y_train)
        model_random_forest = random_forest.fit(X_train_imputed, y_train)

        models = [model_naive_bayes, model_k_neighbors, model_logistic_regression, model_mlp, model_svc, model_decision_tree, model_random_forest]

        for model in models:
            accuracy = model.score(X_test, y_test)
            print(str(model) + ' accuracy: {:.3f}'.format(accuracy))
    
    elif mode == 1:
        # Define a list of n_features_to_select values to try
        n_features_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 ] 
        classifiers = [RandomForestClassifier()]

        # Loop over the different classifiers
        for clf in classifiers:
            print("Classifier:", clf.__class__.__name__)

            # Loop over the different n_features_to_select values
            for n_features in n_features_list:
                # Create an RFE object with the given n_features_to_select value
                rfe = RFE(estimator=clf, n_features_to_select=n_features)

                # Fit the RFE object to the training data
                rfe.fit(X_train, y_train)

                # Select the top features based on the RFE object
                X_train_rfe = rfe.transform(X_train)
                X_test_rfe = rfe.transform(X_test)

                # Train a logistic regression model on the selected features
                model = clf
                model.fit(X_train_rfe, y_train)

                # Evaluate the accuracy on the test data
                accuracy = model.score(X_test_rfe, y_test)

                # Print the accuracy for this iteration
                print("n_features_to_select={}, accuracy={}".format(n_features, accuracy))
    
    elif mode == 2:
        RandomForestClassifier_HyperParameters(X_train, y_train, X_test, y_test)
        Gaussian_Hyperparameters(X_train, y_train, X_test, y_test)
        KNN_Hyperparameters(X_train, y_train, X_test, y_test)
        LogisticRegression_HyperParameters(X_train, y_train, X_test, y_test)
        DecisionTreeClassifier_HyperParameters(X_train, y_train, X_test, y_test)
        SVC_HyperParameters(X_train, y_train, X_test, y_test)
        MLP_HyperParameters(X_train, y_train, X_test, y_test)

    else:
        print("Unknown mode")

    return