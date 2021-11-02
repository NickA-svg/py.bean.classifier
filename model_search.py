from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def best_model(X_train, y_train, X_test,y_test, algo):

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    model = algo()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    training_score = model.score(X_train, y_train)
    acc = accuracy_score(y_test, predictions)
    con = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f'Training Score: {training_score}')
    print(f'Accuracy Score: {acc}')
    print(f'Confusion Matrix: {con}')
    print(f'Classification Report: {report}')
    return acc, training_score