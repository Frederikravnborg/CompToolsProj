# Define the path to the file
file_path = "/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms+spam+collection/SMSSpamCollection.txt"

# Initialize lists to store the tags and texts
tags = []
texts = []

# Open and read the file
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into tag and text
            parts = line.strip().split('\t', 1)  # Split only on the first tab
            if len(parts) == 2:  # Ensure valid format
                tag, text = parts
                tags.append(tag)
                texts.append(text)
except FileNotFoundError:
    print(f"Error: The file at {file_path} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

# Display the first few entries as a preview
print("Tags and Texts Loaded:")
for i in range(min(5, len(tags))):  # Display only the first 5 entries
    print(f"Tag: {tags[i]}, Text: {texts[i]}")

# Save the messages without labels to a new text file
output_text_path = "/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/messages_only.txt"
try:
    with open(output_text_path, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text + '\n')
    print(f"\nMessages without labels saved to {output_text_path}")
except Exception as e:
    print(f"An error occurred while saving the text file: {e}")

# Optionally, save the results into a structured format like a DataFrame
import pandas as pd

data = pd.DataFrame({'Tag': tags, 'Text': texts})

# Show the DataFrame for verification
print("\nDataFrame Preview:")
print(data.head())

# Save to a CSV file if needed
output_csv_path = "/Users/fredmac/Documents/DTU-FredMac/CompTools/CompToolsProj/sms_spam_collection.csv"
try:
    data.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\nData saved to {output_csv_path}")
except Exception as e:
    print(f"An error occurred while saving the CSV file: {e}")