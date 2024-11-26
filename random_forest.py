import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# ham = 0, spam = 1

train_data = pd.read_csv('sms_data/train.csv')
test_data = pd.read_csv('sms_data/test.csv')

vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()

X_train = train_data['Text']
X_train.str.lower()
X_train = vectorizer.fit_transform(X_train)
y_train = label_encoder.fit_transform(train_data['Tag'])


X_test = test_data['Text']
X_test.str.lower()
X_test = vectorizer.fit_transform(X_test)
y_test = label_encoder.fit_transform(test_data['Tag'])

print(X_test)
print(X_train)

model = RandomForestClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, predictions))

