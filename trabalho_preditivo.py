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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
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

    plt.figure()

    models = [
        {
            'label': 'MLP',
            'model': MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='adam', learning_rate='adaptive', learning_rate_init=0.01,
                        max_iter=1000, momentum=0.9, verbose=False, early_stopping=False, n_iter_no_change=10),
        },
        {
            'label': 'RF',
            'model': RandomForestClassifier(n_estimators=10, criterion='entropy', max_features=5),
        },
        {
            'label': 'DT',
            'model': DecisionTreeClassifier(criterion='entropy', splitter='best', max_features=5),
        },
        {
            'label': 'SVM-RBF',
            'model': SVC(C = 1, kernel='rbf', gamma='scale', tol=1e-5, max_iter=1000, probability=True),
        },
        {
            'label': 'LR',
            'model': LogisticRegression(tol=1e-7, solver='liblinear', max_iter=1000),
        }
    ]

    for m in models:
        accuracies = []
        clf = m['model']
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

        fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(x_test)[:,1])
        auc = roc_auc_score(y_test, clf.predict(x_test))
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
    
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    

    # clf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features=5)
    # clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_features=5)

    # clf = SVC(C = 1, kernel='rbf', gamma='scale', tol=1e-5, max_iter=1000, probability=True)

    # clf = LogisticRegression(tol=1e-7, solver='liblinear', max_iter=1000)
    # accuracies = []

    # x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, stratify = y)
    # skf = StratifiedKFold(n_splits=10)
    
    # for train_idx, test_idx in skf.split(x_train, y_train):
    #     X_train, X_test = x_train[train_idx], x_train[test_idx]
    #     Y_train, Y_test = y_train[train_idx], y_train[test_idx]

    #     clf = clf.fit(X_train, Y_train)
    #     accuracies.append(clf.score(X_test, Y_test))
    #     y_pred = clf.predict(x_test)

    # print("Mean accuracy: {0:.3f} (+/- {1:.3f})\n".format(stat.mean(accuracies), stat.stdev(accuracies)))
    # print("ACC: {0:.3f}".format(clf.score(x_test, y_test)))
    # print("F1 Score: {0:.3f}".format(metrics.f1_score(y_test, y_pred)))
    # print("AUC score: {0:.3f}".format((metrics.roc_auc_score(y_test, y_pred))))
    # print("Confusion Matrix:\n{0}".format(metrics.confusion_matrix(y_test, y_pred)))

    # # compute ROC curves
    # ns_probs = [0 for _ in range(len(y_test))]
    # fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
    # ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    # # plot the roc curve for the model
    # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Aleatório')
    # plt.plot(fpr, tpr, marker='.', label='Regressão Logística')
    # # axis labels
    # plt.xlabel('Taxa de Falsos Positivos')
    # plt.ylabel('Taxa de Verdadeiros Positivos')
    # # show the legend
    # plt.legend()
    # # show the plot
    # plt.show()

    

    # skplt.metrics.plot_roc_curve(y_test, clf.predict_proba(x_test))
    # plt.show()

