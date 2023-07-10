from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

iris = load_iris()



X = iris.data
Y = iris.target

#split the dataset into training and testing data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state =42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train,Y_train)

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)

print("Accuracy:",accuracy)

print("confusion matrix:")
print(confusion_matrix(Y_test,y_pred))

print("Classification report:")
print(classification_report(Y_test,y_pred))
