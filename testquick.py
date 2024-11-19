import pandas as pd

def count_tags(file_path):
    """
    Counts the number of 'spam' and 'ham' tags in the given CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary with counts of 'spam' and 'ham'.
    """
    try:
        df = pd.read_csv(file_path)
        counts = df['Tag'].value_counts().to_dict()
        # Ensure both 'spam' and 'ham' are in the dictionary
        counts.setdefault('ham', 0)
        counts.setdefault('spam', 0)
        return counts
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {}
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return {}
    except KeyError:
        print(f"Error: The file '{file_path}' does not contain a 'Tag' column.")
        return {}

def main():
    # Paths to your training and test CSV files
    train_file = 'train.csv'
    test_file = 'test.csv'

    # Count tags in training set
    train_counts = count_tags(train_file)
    if train_counts:
        print(f"Counts in '{train_file}':")
        print(f"  Ham: {train_counts.get('ham', 0)}")
        print(f"  Spam: {train_counts.get('spam', 0)}\n")
        print(train_counts.get('ham', 0)/train_counts.get('spam', 0))

    # Count tags in test set
    test_counts = count_tags(test_file)
    if test_counts:
        print(f"Counts in '{test_file}':")
        print(f"  Ham: {test_counts.get('ham', 0)}")
        print(f"  Spam: {test_counts.get('spam', 0)}")
        print(test_counts.get('ham', 0)/test_counts.get('spam', 0))

if __name__ == "__main__":
    main()