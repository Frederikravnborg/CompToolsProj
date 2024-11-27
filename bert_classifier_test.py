import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score  # Added for Balanced Accuracy
)
import os
from tqdm import tqdm

# ---------------------------
# 1. Configuration and Setup
# ---------------------------

# Set seed for reproducibility (optional)
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
MODEL_SAVE_PATH = 'spam_classifier_model'  # Directory where the trained model is saved
TEST_DATA_PATH = "/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_data/test.csv"    # Path to test data
MAX_LEN = 128                           # Maximum length of input sequences
BATCH_SIZE = 16                         # Batch size for evaluation
EVALUATION_SAVE_DIR = 'evaluation_results'  # Directory to save evaluation results
EVALUATION_RESULTS_FILE = 'evaluation_results.txt'  # Single text file for all evaluation outputs

# ---------------------------
# 2. Load the Trained Model
# ---------------------------

# Verify that the model directory exists
if not os.path.exists(MODEL_SAVE_PATH):
    raise FileNotFoundError(f"Model directory '{MODEL_SAVE_PATH}' does not exist. Please check the path.")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

print("Model and tokenizer loaded successfully.")

# ---------------------------
# 3. Load and Preprocess the Test Data
# ---------------------------

# Load the test dataset
test_df = pd.read_csv(TEST_DATA_PATH)

# Display basic information about the test dataset
print(f'Test dataset contains {len(test_df)} samples.')
print(test_df.head())

# Encode labels: ham=0, spam=1
label_mapping = {'ham': 0, 'spam': 1}
test_df['Tag'] = test_df['Tag'].map(label_mapping)

# Verify label encoding
print("\nTest label distribution:")
print(test_df['Tag'].value_counts())

# Extract texts and labels
test_texts = test_df['Text'].values
test_labels = test_df['Tag'].values

# Define a custom Dataset class
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # Add [CLS] and [SEP]
            max_length=self.max_len,
            padding='max_length',         # Pad to max_length
            truncation=True,              # Truncate to max_length
            return_attention_mask=True,
            return_tensors='pt',          # Return PyTorch tensors
        )
        
        input_ids = encoding['input_ids'].squeeze()           # Shape: (max_len)
        attention_mask = encoding['attention_mask'].squeeze() # Shape: (max_len)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create Dataset and DataLoader instances
test_dataset = SpamDataset(test_texts, test_labels, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Test data loaded and preprocessed successfully.")

# ---------------------------
# 4. Perform Predictions on Test Data
# ---------------------------

all_preds = []
all_labels = []
all_probs = []

print("Starting predictions on test data...")

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        
        logits = outputs.logits  # Shape: (batch_size, num_labels)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of the positive class (spam)
        preds = torch.argmax(logits, dim=1)
        
        # Move tensors to CPU and convert to numpy
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

print("Predictions completed.")

# ---------------------------
# 5. Compute Evaluation Metrics
# ---------------------------

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)
balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)  # Computed Balanced Accuracy
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=['Ham', 'Spam'])

# Display the metrics
print("\n=== Evaluation Metrics ===")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Balanced Accuracy  : {balanced_accuracy:.4f}")  # Printed Balanced Accuracy
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1-Score           : {f1:.4f}")
print(f"ROC-AUC            : {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# ---------------------------
# 6. Save Evaluation Results
# ---------------------------

# Create the evaluation save directory if it doesn't exist
if not os.path.exists(EVALUATION_SAVE_DIR):
    os.makedirs(EVALUATION_SAVE_DIR)

# Define the path for the single evaluation results file
evaluation_results_path = os.path.join(EVALUATION_SAVE_DIR, EVALUATION_RESULTS_FILE)

# Open the file in write mode
with open(evaluation_results_path, 'w') as f:
    # Write Metrics
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"Accuracy           : {accuracy:.4f}\n")
    f.write(f"Balanced Accuracy  : {balanced_accuracy:.4f}\n")
    f.write(f"Precision          : {precision:.4f}\n")
    f.write(f"Recall             : {recall:.4f}\n")
    f.write(f"F1-Score           : {f1:.4f}\n")
    f.write(f"ROC-AUC            : {roc_auc:.4f}\n\n")
    
    # Write Confusion Matrix
    f.write("=== Confusion Matrix ===\n")
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=['Actual_Ham', 'Actual_Spam'],
        columns=['Predicted_Ham', 'Predicted_Spam']
    )
    f.write(conf_matrix_df.to_string())
    f.write("\n\n")
    
    # Write Classification Report
    f.write("=== Classification Report ===\n")
    f.write(class_report)
    
print(f"All evaluation results saved to '{evaluation_results_path}'.")