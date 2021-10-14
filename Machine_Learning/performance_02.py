import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, classification_report
from Fetch_data import fetch_data


data_path = 'C:/Users/Jurg Verhoeven/Documents/Cook3r/Testds'

# Fetch the data
foods = fetch_data(data_path)

# Sample a training set while holding out 40% of the data for testing (evaluating) our classifier
X_train, X_test, y_train, y_test = train_test_split(
    foods.data, foods.target, test_size=0.4, random_state=0)

# instantiate a classifier estimator
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])

# fit the classifier
clf.fit(X_train, y_train)

# Plot confusion matrix
fig0, ax0 = plt.subplots(1,1)
plot_confusion_matrix(clf, X_test, y_test, ax=ax0)
ax0.set_title('Confusion matrix')
plt.tight_layout()

# Show detailed classification report
y_true, y_pred = y_test, clf.predict(X_test)
print("Detailed classification report:")
print()
print(classification_report(y_true, y_pred))
print()


plt.show(block=True)
