import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split  # Added for data splitting
from tqdm import tqdm
import os

# ---------------------------
# 1. Configuration and Setup
# ---------------------------

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check if GPU is available
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
MAX_LEN = 128                       # Maximum length of input sequences
BATCH_SIZE = 16                     # Batch size for training and validation
EPOCHS = 10                         # Updated number of training epochs
MODEL_NAME = 'bert-base-uncased'    # Pre-trained BERT model
TRAIN_DATA_PATH = "/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_data/train.csv"  # Path to training data
TEST_DATA_PATH = "/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_data/test.csv"    # Path to test data
MODEL_SAVE_PATH = 'spam_classifier_model'  # Directory to save the trained model

# ---------------------------------
# 2. Load and Preprocess the Data
# ---------------------------------

# Load the training dataset
train_df = pd.read_csv(TRAIN_DATA_PATH)

# Load the test dataset
test_df = pd.read_csv(TEST_DATA_PATH)

# Display basic information about the datasets
print(f'Training dataset contains {len(train_df)} samples.')
print(f'Test dataset contains {len(test_df)} samples.')
print(train_df.head())

# Encode labels: ham=0, spam=1
label_mapping = {'ham': 0, 'spam': 1}
train_df['Tag'] = train_df['Tag'].map(label_mapping)
test_df['Tag'] = test_df['Tag'].map(label_mapping)

# Verify label encoding
print("Training label distribution:")
print(train_df['Tag'].value_counts())
print("\nTest label distribution:")
print(test_df['Tag'].value_counts())

# Split the training data into training and validation sets (90% train, 10% val)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['Text'].values,
    train_df['Tag'].values,
    test_size=0.1,
    random_state=42,
    stratify=train_df['Tag'].values  # Ensure the split maintains class distribution
)

print(f'\nAfter splitting:')
print(f'Training set contains {len(train_texts)} samples.')
print(f'Validation set contains {len(val_texts)} samples.')

# ---------------------------------
# 3. Handle Class Imbalance
# ---------------------------------

# Compute class weights based on the training data
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f'Class weights: {class_weights}')

# ---------------------------------
# 4. Tokenization
# ---------------------------------

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

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

# Create Dataset instances
train_dataset = SpamDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SpamDataset(val_texts, val_labels, tokenizer, MAX_LEN)  # Validation dataset
test_dataset = SpamDataset(test_df['Text'].values, test_df['Tag'].values, tokenizer, MAX_LEN)  # Test dataset

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)      # Validation DataLoader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)    # Test DataLoader

# ---------------------------------
# 5. Initialize the Model
# ---------------------------------

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(device)

# ---------------------------------
# 6. Define Optimizer and Scheduler
# ---------------------------------

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Total number of training steps
total_steps = len(train_loader) * EPOCHS

# Define the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,                # Default value
    num_training_steps=total_steps
)

# ---------------------------------
# 7. Define the Loss Function
# ---------------------------------

# Use CrossEntropyLoss with class weights to handle imbalance
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# ---------------------------------
# 8. Training and Validation Loops
# ---------------------------------

best_val_accuracy = 0
best_model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pt')

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    # -----------------
    # Training Phase
    # -----------------
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        
        loss = outputs.loss
        train_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f'Training loss: {avg_train_loss:.4f}')
    
    # -----------------
    # Validation Phase
    # -----------------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            val_loss += loss.item()
            
            # Calculate predictions
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    print(f'Validation loss: {avg_val_loss:.4f}')
    print(f'Validation accuracy: {val_accuracy:.4f}')
    
    # Check if this is the best model so far; if so, save it
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        # Save the model's state_dict
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model updated and saved with accuracy: {best_val_accuracy:.4f}')

# ---------------------------------
# 9. Save the Trained Model
# ---------------------------------

# Ensure the save directory exists
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Save the tokenizer
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f'\nTokenizer saved to {MODEL_SAVE_PATH}')

# ---------------------------------
# 10. Load the Best Model and Evaluate on Test Set
# ---------------------------------

# Load the best model
best_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
best_model.load_state_dict(torch.load(best_model_path))
best_model = best_model.to(device)
best_model.eval()
print('\nBest model loaded for testing.')

# Evaluate on the test set
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = best_model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        
        loss = outputs.loss
        test_loss += loss.item()
        
        # Calculate predictions
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = correct / total
print(f'\nTest loss: {avg_test_loss:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')