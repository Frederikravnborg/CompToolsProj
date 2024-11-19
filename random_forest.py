import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# ham = 0, spam = 1



train_data = pd.read_csv('sms_data/train.csv')
test_data = pd.read_csv('sms_data/test.csv')


vectorizer = TfidfVectorizer(max_features=5000)
label_encoder = LabelEncoder()
X_train = label_encoder.fit_transform(train_data['Tag'])
X_train = X_train.reshape(-1,1)
y_train = train_data['Text']
y_train.str.lower()


X_test = label_encoder.fit_transform(test_data['Tag'])
X_test = X_test.reshape(-1,1)
y_test = test_data['Text']

#data['text'] = data['text'].str.lower()  # Lowercase
#X = vectorizer.fit_transform(data['text'])

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, predictions))




