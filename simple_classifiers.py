from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

test = pd.read_csv('/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_data/test.csv')
train = pd.read_csv('/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_data/train.csv')
X_train = train['Text']
y_train = train['Tag']
X_test = test['Text']
y_test = test['Tag']

# Define classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'NaiveBayes': MultinomialNB()
}

# Open a single results file in write mode before the loop
with open('results_all_classifiers.txt', 'w') as f:
    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        print(f'Classifier: {name}')
        print(f'Balanced Accuracy: {balanced_acc}')
        print(classification_report(y_test, predictions))
    
        f.write(f'Classifier: {name}\n')
        f.write(f'Balanced Accuracy: {balanced_acc}\n')
        f.write(classification_report(y_test, predictions))
        f.write('\n')
