#!/usr/bin/env python3
import os
import pandas as pd
import shutil

# Set paths to your CSV file and the source directory of text files.
csv_file = '/Users/finnsommer/llama-finetune/Data/Gutenberg Data/pg_catalog.csv'
books_folder = '/Users/finnsommer/llama-finetune/Data/Gutenberg Data/Books_txt_Files'

# Define the destination folder
# Mark Twain
# Uncomment to use

# destination_folder = '/Users/finnsommer/llama-finetune/Data/Mark_Twain'

# Charles Dickens
# Uncomment to use

destination_folder = '/Users/finnsommer/llama-finetune/Data/Charles_Dickens'


# Create the destination folder if it does not exist.
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created destination folder: {destination_folder}")

# Read the CSV file into a pandas DataFrame.
df = pd.read_csv(csv_file)

if(destination_folder.endswith("Charles_Dickens")):
    # Filter for rows that have "Dickens, Charles" in the Authors column and language "en".
    # This check is case-sensitive; adjust by using .str.lower() if needed.
    filtered_df = df[(df['Authors'].str.contains("Twain, Mark", na=False)) & (df['Language'].str.lower() == "en")]

    print(f"Found {len(filtered_df)} text entries for Mark Twain in English.")  
elif(destination_folder.endswith("Mark_Twain")):
    # Filter for rows that have "Twain, Mark" in the Authors column and language "en".
    # This check is case-sensitive; adjust by using .str.lower() if needed.
    filtered_df = df[(df['Authors'].str.contains("Twain, Mark", na=False)) & (df['Language'].str.lower() == "en")]

print(f"Found {len(filtered_df)} text entries for Mark Twain in English.")

# Loop over each row in the filtered DataFrame.
for index, row in filtered_df.iterrows():
    text_num = str(row['Text#']).strip()  # Get the text number as a string.
    # Build the source file path based on the assumed structure.
    # For example: /Users/finnsommer/llama-finetune/Data/Books_txt_Files/98/pg98.txt
    source_file = os.path.join(books_folder, text_num, f"pg{text_num}.txt")
    
    # Check if the file exists
    if os.path.exists(source_file):
        # Define the destination file path (keeping the same file name).
        dest_file = os.path.join(destination_folder, f"pg{text_num}.txt")
        try:
            shutil.copy2(source_file, dest_file)
            print(f"Copied {source_file} to {dest_file}.")
        except Exception as e:
            print(f"Error copying {source_file} to {dest_file}: {e}")
    else:
        print(f"Source file not found: {source_file}")

print("File copying complete.")
