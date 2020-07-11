import pandas as pd
import numpy as np
import statistics as stat
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import scikitplot as skplt
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_excel("dados_oswaldo_preditivo.xlsx")
    df.replace('N', 0, True)
    df.replace('S', 1, True)
    
    x = df.loc[:, 'A1':'A8'].values
    y = df[["Classe"]].values

    x_norm = StandardScaler().fit_transform(x)
    pca = PCA(n_components=8)
    x_pca = pca.fit_transform(X = x_norm)
    
    # clf = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='adam', learning_rate='adaptive', learning_rate_init=0.01,
    #                     max_iter=1000, momentum=0.9, verbose=False, early_stopping=False, n_iter_no_change=10)

    # clf = RandomForestClassifier(n_estimators=9, criterion='entropy')
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='auto')
    # clf = LogisticRegression(tol=1e-7, solver='liblinear', max_iter=1000)
    accuracies = []

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, stratify = y)
    skf = StratifiedKFold(n_splits=10)
    
    for train_idx, test_idx in skf.split(x_train, y_train):
        X_train, X_test = x_train[train_idx], x_train[test_idx]
        Y_train, Y_test = y_train[train_idx], y_train[test_idx]

        clf = clf.fit(X_train, Y_train)
        accuracies.append(clf.score(X_test, Y_test))
        y_pred = clf.predict(x_test)

    print("Mean accuracy: {0:.3f} (+/- {1:.3f})\n".format(stat.mean(accuracies), stat.stdev(accuracies)))
    print("ACC: {0:.3f}".format(clf.score(x_test, y_test)))
    print("F1 Score: {0:.3f}".format(metrics.f1_score(y_test, y_pred)))
    print("AUC score: {0:.3f}".format((metrics.roc_auc_score(y_test, y_pred))))
    print("Confusion Matrix:\n{0}".format(metrics.confusion_matrix(y_test, y_pred)))
    skplt.metrics.plot_roc_curve(y_test, clf.predict_proba(x_test))
    plt.show()

    # print(np.count_nonzero(x_test == 0))
    # print(np.count_nonzero(x_test == 1))


