import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS backend.")
else:
    device = torch.device('cpu')
    print("MPS backend not available. Using CPU.")

# Load your predefined training and testing datasets
# Replace 'train_dataset.csv' and 'test_dataset.csv' with the paths to your datasets
train_df = pd.read_csv('sms_data/train.csv')
test_df = pd.read_csv('sms_data/test.csv')

# Extract texts and labels from the training set
X_train_texts = train_df['Text'].tolist()
y_train = train_df['Tag'].tolist()

# Extract texts and labels from the test set
X_test_texts = test_df['Text'].tolist()
y_test = test_df['Tag'].tolist()

# Convert labels to numerical format if they are not already
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # Now y_train will be numerical labels
y_test = label_encoder.transform(y_test)        # Ensure consistent label encoding

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to get BERT embeddings
def get_bert_embedding(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    # Get the outputs from BERT
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Get the pooled output embeddings
    embeddings = outputs.pooler_output.cpu().numpy()  # Move embeddings to CPU
    return embeddings

# Generate embeddings for the training set
print("Generating BERT embeddings for the training set...")
train_embeddings = []
for idx, text in enumerate(X_train_texts):
    emb = get_bert_embedding(text)
    train_embeddings.append(emb)
    if (idx + 1) % 100 == 0 or idx + 1 == len(X_train_texts):
        print(f"Processed {idx + 1}/{len(X_train_texts)} training texts")

train_embeddings = np.vstack(train_embeddings)

# Generate embeddings for the test set
print("Generating BERT embeddings for the test set...")
test_embeddings = []
for idx, text in enumerate(X_test_texts):
    emb = get_bert_embedding(text)
    test_embeddings.append(emb)
    if (idx + 1) % 100 == 0 or idx + 1 == len(X_test_texts):
        print(f"Processed {idx + 1}/{len(X_test_texts)} test texts")

test_embeddings = np.vstack(test_embeddings)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
print("Training the Random Forest classifier...")
clf.fit(train_embeddings, y_train)

# Predict on the test set
print("Making predictions on the test set...")
y_pred = clf.predict(test_embeddings)

# Evaluate the model
print("Evaluating the model...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced accuracy score:", balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
