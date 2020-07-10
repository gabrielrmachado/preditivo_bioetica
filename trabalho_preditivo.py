import pandas as pd
import numpy as np
import statistics as stat
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


if __name__ == "__main__":
    df = pd.read_excel("dados_oswaldo_preditivo.xlsx")
    df.replace('N', 0, True)
    df.replace('S', 1, True)
    
    x = df.loc[:, 'A1':'A8'].values
    y = df[["Classe"]].values

    # x_norm = StandardScaler().fit_transform(x)

    # pca = PCA(n_components=8)
    # x_pca = pca.fit_transform(X = x)
    # print(x_norm)
    
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='sgd', learning_rate='adaptive', 
                        learning_rate_init=0.01, max_iter=500, momentum=0.9, tol=1e-5)
    accuracies = []

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify = y)
    skf = StratifiedKFold(n_splits=10)
    
    for train_idx, test_idx in skf.split(x_train, y_train):
        X_train, X_test = x_train[train_idx], x_train[test_idx]
        Y_train, Y_test = y_train[train_idx], y_train[test_idx]

        clf.fit(X_train, Y_train)
        accuracies.append(clf.score(X_test, Y_test))

    print("Mean accuracy: {0:.3f} (+/- {1:.3f})\n".format(stat.mean(accuracies), stat.stdev(accuracies)))
    print(clf.score(x_test, y_test))
    print(metrics.roc_auc_score(x_test, y_test))



    # print(np.count_nonzero(x_test == 0))
    # print(np.count_nonzero(x_test == 1))


