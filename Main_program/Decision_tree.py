from Food_recognition import Food_recognition
import Pan
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import Food
import os
from Fetch_data import fetch_data
from sklearn import tree
import graphviz 
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix, classification_report
from mlxtend.evaluate import bias_variance_decomp


food_path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_v2"


if __name__ == "__main__":

    data_path = 'C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_v2'

    # Fetch the data
    # foods = fetch_data(food_path)

    # dump(foods, "foods.joblib")

    foods = load("foods.joblib")

    # encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(foods.target)

    print(coded_labels)

    # partition the data into training and testing splits using 60% of
    # the data for training and the remaining 40% for testing
    (trainX, testX, trainY, testY) = train_test_split(foods.data, coded_labels,
	test_size=0.40, stratify=foods.target)

    # Data preparation (note that a pipeline  would help here)
    # trainX = StandardScaler().fit_transform(trainX)

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(trainX, trainY)

    dump(clf, "decision_tree_model.joblib")

    plt.figure()
    tree.plot_tree(clf,filled=True)  
    plt.savefig('tree.svg',format='svg',bbox_inches = "tight")


    # tree.plot_tree(clf)

   # Plot confusion matrix
    fig0, ax0 = plt.subplots(1,1)
    plot_confusion_matrix(clf, testX, testY, ax=ax0)
    ax0.set_title('Confusion matrix')
    plt.tight_layout()

    # Show detailed classification report
    y_true, y_pred = testY, clf.predict(testX)
    print("Detailed classification report:")
    print()
    print(classification_report(y_true, y_pred))
    print()
    plt.show(block=True)

    mse, bias, var = bias_variance_decomp(clf, trainX, trainY, testX, testY, loss='mse', num_rounds=200, random_seed=1)
    # summarize results
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)