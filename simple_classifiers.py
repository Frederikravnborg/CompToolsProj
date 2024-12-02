from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

# Load data
test = pd.read_csv('/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_data/test.csv')
train = pd.read_csv('/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_data/train.csv')
X_train = train['Text']
y_train = train['Tag']
X_test = test['Text']
y_test = test['Tag']


class_weights = {'ham': 0.5774, 'spam': 3.7306}

# Define classifiers with class_weight where applicable
classifiers = {
    'RandomForest': RandomForestClassifier(class_weight=class_weights, random_state=42),
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),  # KNN does not support class_weight
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),  # GradientBoosting does not support class_weight directly
    'NaiveBayes': MultinomialNB()  # MultinomialNB does not support class_weight
}

# Open a single results file in write mode before the loop
with open('results_all_classifiers.txt', 'w') as f:
    for name, clf in classifiers.items():
        # Define the pipeline
        pipeline_steps = [
            ('tfidf', TfidfVectorizer()),
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', clf)
        ]
        
        # For classifiers that do not support class_weight, consider alternative handling (e.g., resampling)
        # You can implement additional logic here if desired
        
        pipeline = Pipeline(pipeline_steps)
        
        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)
        
        # Make predictions on the test data
        predictions = pipeline.predict(X_test)
        
        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        
        # Print and write the classification report
        print(f'Classifier: {name}')
        print(f'Balanced Accuracy: {balanced_acc}')
        print(classification_report(y_test, predictions))
        
        f.write(f'Classifier: {name}\n')
        f.write(f'Balanced Accuracy: {balanced_acc}\n')
        f.write(classification_report(y_test, predictions))
        f.write('\n')