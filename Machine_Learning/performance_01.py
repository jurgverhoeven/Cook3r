import cv2 as cv
from Fetch_data import fetch_data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix, classification_report



data_path = 'C:/Users/Jurg Verhoeven/Documents/Cook3r/Testds'

# Fetch the data
foods = fetch_data(data_path)

cv.destroyAllWindows()

# Sample a training set while holding out 25% of the data for testing (evaluating) our classifier
X_train, X_test, y_train, y_test = train_test_split(
    foods.data, foods.target, test_size=0.25, random_state=0)

# instantiate a classifier estimator
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])

# fit the classifier
clf.fit(X_train, y_train)

# show the result of the scoring method of the classifier. For SVC, the score is mean accuracy
print("mean accuracy on the given test data and labels: {}".format(clf.score(X_test, y_test)))

# compute some other metric to evaluate the model
scoring_metric = "precision_macro"
nr_of_folds = 5
score = cross_val_score(clf, X_train, y_train, cv=nr_of_folds, scoring=scoring_metric)
print("{}-fold cross validation metric {} : {}".format(nr_of_folds, scoring_metric, score))

