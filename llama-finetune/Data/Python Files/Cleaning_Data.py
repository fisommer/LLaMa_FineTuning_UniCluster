#!/usr/bin/env python3
import os
import glob
import re
import shutil

# Define source and destination folders.

# Charles Dickens
# Comment to use
#'''
src_folder = "/Users/finnsommer/llama-finetune/Data/Charles_Dickens"
dest_folder = "/Users/finnsommer/llama-finetune/Data/Charles_Dickens/Cleaned"
# '''

# Mark Twain
# Comment to use
'''
src_folder = "/Users/finnsommer/llama-finetune/Data/Mark_Twain"
dest_folder = "/Users/finnsommer/llama-finetune/Data/Mark_Twain/Cleaned"
# '''

# Create the destination folder if it doesn't exist.
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Find all .txt files recursively in the source folder.
files = glob.glob(os.path.join(src_folder, "**", "*.txt"), recursive=True)

# Define markers and regex for extracting the title.
start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
end_marker = "*** END OF THE PROJECT GUTENBERG"
start_pattern = re.compile(
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK\s*(.*?)\s*\*\*\*",
    re.IGNORECASE | re.DOTALL
)

# Process each file.
for file_path in files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read file {file_path}: {e}")
        continue

    start_index = text.find(start_marker)
    if start_index == -1:
        print(f"[WARNING] Start marker not found in file {file_path}. Skipping file.")
        continue

    match = start_pattern.search(text)
    if match:
        title = match.group(1).strip()
    else:
        print(f"[WARNING] Could not extract title using regex in file {file_path}. Skipping file.")
        continue

    # Sanitize the title for safe filename usage.
    safe_title = re.sub(r"[^\w\s-]", "", title).strip()
    safe_title = re.sub(r"\s+", "_", safe_title)
    if not safe_title:
        print(f"[WARNING] Title after sanitization is empty for file {file_path}. Skipping file.")
        continue

    # Remove all text before the header line ends.
    header_line_end = text.find("\n", start_index)
    if header_line_end == -1:
        header_line_end = start_index + len(start_marker)
    else:
        header_line_end += 1  # Skip the newline.

    # Cleaned text starts after the header line.
    cleaned_text = text[header_line_end:]

    # Find the end marker and truncate the text, if found.
    end_index = cleaned_text.find(end_marker)
    if end_index != -1:
        cleaned_text = cleaned_text[:end_index]
    else:
        print(f"[WARNING] End marker not found in file {file_path}. Keeping all text from header onward.")

    # Define destination file path using the safe title.
    dest_file = os.path.join(dest_folder, f"{safe_title}.txt")
    try:
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
    except Exception as e:
        print(f"[ERROR] Could not write cleaned file {dest_file}: {e}")
        continue

    # Delete the original file.
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"[WARNING] Could not delete original file {file_path}: {e}")

print("Cleaning process complete.")
