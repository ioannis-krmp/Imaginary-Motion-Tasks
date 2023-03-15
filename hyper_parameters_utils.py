import os
import time
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from multiprocessing import cpu_count
from sklearn.pipeline import Pipeline

def Gaussian_Hyperparameters(X_train, y_train, X_test, y_test):
    print("-----------------------------------------------------------------------------------------")
    start = time.time()

    gnb = GaussianNB()

    hyperparameters = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    n_jobs = cpu_count() if not 'X_IGNORE_LAUNCH_CMD' in os.environ else 1
    clf = GridSearchCV(gnb, hyperparameters, cv=5, n_jobs = n_jobs)
    clf.fit(X_train, y_train)
    end = time.time()
    minutes, remaining_seconds = divmod(end - start, 60)
    print("Fitting of gaussian took {} minutes and {} seconds".format(minutes, remaining_seconds))
    # print the best hyperparameters and accuracy
    #print("Best hyperparameters:", clf.best_params_)
    print("Accuracy:", clf.best_score_)
    print("-----------------------------------------------------------------------------------------")
    return

def KNN_Hyperparameters(X_train, y_train, X_test, y_test):
    print("-----------------------------------------------------------------------------------------")
    start = time.time()

    k_neighbors = KNeighborsClassifier()
    param_grid_k_neighbors = {  
                                #'pca__n_components': [0.95, 0.90, 0.85, 0.80],
                                'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
                                'kneighborsclassifier__weights': ['uniform', 'distance'],
                                'kneighborsclassifier__metric': ['euclidean', 'manhattan']
                            }
    scorer_k_neighbors = make_scorer(f1_score, average = 'macro')

    n_jobs = cpu_count() if not 'X_IGNORE_LAUNCH_CMD' in os.environ else 1
    #print(n_jobs)

    pipeline_k_neighbors = Pipeline([
        ('kneighborsclassifier', k_neighbors)
    ])

    grid_search_scorer_k_neighbors = GridSearchCV(pipeline_k_neighbors, param_grid_k_neighbors, scoring=scorer_k_neighbors, cv=5, n_jobs=n_jobs)
    grid_search_scorer_k_neighbors.fit(X_train, y_train)

    end = time.time()
    minutes, remaining_seconds = divmod(end - start, 60)
    print("Fitting of k_neighbors took {} minutes and {} seconds".format(minutes, remaining_seconds))
    # Get the best hyperparameters found by the grid search
    best_params_k_neighbors = grid_search_scorer_k_neighbors.best_params_
    best_score_k_neighbors = grid_search_scorer_k_neighbors.best_score_
    best_estimator_k_neighbors = grid_search_scorer_k_neighbors.best_estimator_
    test_accuracy_k_neighbors = best_estimator_k_neighbors.score(X_test, y_test)

    print("Best hyperparameters: ", best_params_k_neighbors)
    #print("Best cross-validated score:", best_score_k_neighbors)
    print("Test accuracy:", test_accuracy_k_neighbors) 
    print("-----------------------------------------------------------------------------------------")
    return

def LogisticRegression_HyperParameters(X_train, y_train, X_test, y_test):
    print("-----------------------------------------------------------------------------------------")
    start = time.time()

    logistic_regression = LogisticRegression()
    param_grid_logistic_regression = {  
                                        #'pca__n_components': [0.95, 0.90, 0.85, 0.80],
                                        'logisticregression__penalty': ['l1', 'l2'],
                                        'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                        'logisticregression__solver': ['liblinear', 'saga'],
                                        'logisticregression__max_iter': [100, 200, 300, 400, 500]

                                    }
    scorer_logistic_regression = make_scorer(f1_score, average = 'macro')

    n_jobs = cpu_count() if not 'X_IGNORE_LAUNCH_CMD' in os.environ else 1
    #print("CPU cores available = ", n_jobs)

    pipeline_logistic_regression = Pipeline([
        ('logisticregression', logistic_regression)
    ])

    grid_search_scorer_logistic_regression = GridSearchCV(pipeline_logistic_regression, param_grid_logistic_regression, scoring=scorer_logistic_regression, cv=5, n_jobs=n_jobs)
    grid_search_scorer_logistic_regression.fit(X_train, y_train)

    end = time.time()
    minutes, remaining_seconds = divmod(end - start, 60)
    print("Fitting of logistic regression took {} minutes and {} seconds".format(minutes, remaining_seconds))
    # Get the best hyperparameters found by the grid search
    best_params_logistic_regression = grid_search_scorer_logistic_regression.best_params_
    best_score_logistic_regression = grid_search_scorer_logistic_regression.best_score_
    best_estimator_logistic_regression = grid_search_scorer_logistic_regression.best_estimator_
    test_accuracy_logistic_regression = best_estimator_logistic_regression.score(X_test, y_test)

    print("Best hyperparameters: ", best_params_logistic_regression)
    #print("Best cross-validated score:", best_score_logistic_regression)
    print("Test accuracy:", test_accuracy_logistic_regression)
    print("-----------------------------------------------------------------------------------------")
    return

def DecisionTreeClassifier_HyperParameters(X_train, y_train, X_test, y_test):
    print("-----------------------------------------------------------------------------------------")
    start = time.time()

    decision_tree = DecisionTreeClassifier()
    param_grid_decision_tree = {  
                                'max_depth': [None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                'min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                'criterion': ['gini', 'entropy']
                            }
    scorer_decision_tree = make_scorer(f1_score, average = 'macro')

    n_jobs = cpu_count() if not 'X_IGNORE_LAUNCH_CMD' in os.environ else 1

    grid_search_scorer_decision_tree = GridSearchCV(decision_tree, param_grid_decision_tree, scoring=scorer_decision_tree, cv=5, n_jobs=n_jobs)
    grid_search_scorer_decision_tree.fit(X_train, y_train)

    end = time.time()
    minutes, remaining_seconds = divmod(end - start, 60)
    print("Fitting of decision tree took {} minutes and {} seconds".format(minutes, remaining_seconds))

    # Get the best hyperparameters found by the grid search
    best_params_decision_tree = grid_search_scorer_decision_tree.best_params_
    best_score_decision_tree = grid_search_scorer_decision_tree.best_score_
    best_estimator_decision_tree = grid_search_scorer_decision_tree.best_estimator_
    test_accuracy_decision_tree = best_estimator_decision_tree.score(X_test, y_test)

    print("Best hyperparameters: ", best_params_decision_tree)
    #print("Best cross-validated score:", best_score_decision_tree)
    print("Test accuracy:", test_accuracy_decision_tree)
    print("-----------------------------------------------------------------------------------------")
    return


def RandomForestClassifier_HyperParameters(X_train, y_train, X_test, y_test):
    print("-----------------------------------------------------------------------------------------")
    start = time.time()

    random_forest = RandomForestClassifier()
    param_grid_random_forest = {
                                    #'pca__n_components': [0.95, 0.90],
                                    'randomforestclassifier__n_estimators': [10, 20, 30, 40, 50],
                                    'randomforestclassifier__max_depth': [None, 5, 10, 15, 20, 25],
                                    'randomforestclassifier__min_samples_split': [2, 4, 6, 8, 10],
                                    'randomforestclassifier__min_samples_leaf': [1, 2, 4, 6, 8, 10],
                                    'randomforestclassifier__max_features': ['sqrt', 'log2']
                                }
    scorer_random_forest = make_scorer(f1_score, average = 'macro')

    n_jobs = cpu_count() if not 'X_IGNORE_LAUNCH_CMD' in os.environ else 1
    #print("CPU cores available = ", n_jobs)

    pipeline_random_forest = Pipeline([
        ('randomforestclassifier', random_forest)
    ])

    grid_search_scorer_random_forest = GridSearchCV(pipeline_random_forest, param_grid_random_forest, scoring=scorer_random_forest, cv=5, n_jobs=n_jobs)
    grid_search_scorer_random_forest.fit(X_train, y_train)

    end = time.time()
    minutes, remaining_seconds = divmod(end - start, 60)
    print("Fitting of random_forest took {} minutes and {} seconds".format(minutes, remaining_seconds))

    best_params_random_forest = grid_search_scorer_random_forest.best_params_
    best_score_random_forest = grid_search_scorer_random_forest.best_score_
    best_estimator_random_forest = grid_search_scorer_random_forest.best_estimator_
    test_accuracy_random_forest = best_estimator_random_forest.score(X_test, y_test)

    print("Best hyperparameters: ", best_params_random_forest)
    #print("Best cross-validated score:", best_score_random_forest)
    print("Test accuracy:", test_accuracy_random_forest)
    print("-----------------------------------------------------------------------------------------")   
    return

def SVC_HyperParameters(X_train, y_train, X_test, y_test):
    print("-----------------------------------------------------------------------------------------")
    start = time.time()

    svc_clf = SVC()
    svc_params = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10]
    }
    scorer_CSV = make_scorer(f1_score, average = 'macro')

    n_jobs = cpu_count() if not 'X_IGNORE_LAUNCH_CMD' in os.environ else 1
    #print("CPU cores available = ", n_jobs)

    grid_search_scorer_CSV= GridSearchCV(svc_clf, svc_params, scoring=scorer_CSV, cv=5, n_jobs=n_jobs)
    grid_search_scorer_CSV.fit(X_train, y_train)

    end = time.time()
    minutes, remaining_seconds = divmod(end - start, 60)
    print("Fitting of SVC took {} minutes and {} seconds".format(minutes, remaining_seconds))

    best_params_CSV = grid_search_scorer_CSV.best_params_
    best_score_CSV = grid_search_scorer_CSV.best_score_
    best_estimator_CSV = grid_search_scorer_CSV.best_estimator_
    test_accuracy_CSV = best_estimator_CSV.score(X_test, y_test)

    print("Best hyperparameters: ", best_params_CSV)
    #print("Best cross-validated score:", best_score_CSV)
    print("Test accuracy:", test_accuracy_CSV)
    print("-----------------------------------------------------------------------------------------")
    return

def MLP_HyperParameters(X_train, y_train, X_test, y_test):
    print("-----------------------------------------------------------------------------------------")
    start = time.time()

    mlp_clf = MLPClassifier()
    mlp_params = {
        'hidden_layer_sizes': [(100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01]
    }
    scorer_MLP = make_scorer(f1_score, average = 'macro')

    n_jobs = cpu_count() if not 'X_IGNORE_LAUNCH_CMD' in os.environ else 1
    #print("CPU cores available = ", n_jobs)

    grid_search_scorer_MLP= GridSearchCV(mlp_clf, mlp_params, scoring=scorer_MLP, cv=5, n_jobs=n_jobs)
    grid_search_scorer_MLP.fit(X_train, y_train)

    end = time.time()
    minutes, remaining_seconds = divmod(end - start, 60)
    print("Fitting of MLP took {} minutes and {} seconds".format(minutes, remaining_seconds))

    best_params_MLP = grid_search_scorer_MLP.best_params_
    best_score_MLP = grid_search_scorer_MLP.best_score_
    best_estimator_MLP = grid_search_scorer_MLP.best_estimator_
    test_accuracy_MLP = best_estimator_MLP.score(X_test, y_test)

    print("Best hyperparameters: ", best_params_MLP)
    #print("Best cross-validated score:", best_score_MLP)
    print("Test accuracy:", test_accuracy_MLP)
    print("-----------------------------------------------------------------------------------------")
    return
