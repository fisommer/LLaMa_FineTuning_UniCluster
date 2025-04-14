#!/usr/bin/env python3
import os
import glob
import re
import shutil

# Define folders.

# Mark Twain
# Comment to use
'''
cleaned_folder = "/Users/finnsommer/llama-finetune/Data/Mark_Twain/Cleaned"
eval_folder = "/Users/finnsommer/llama-finetune/Data/Mark_Twain/Evaluation_Data"
final_folder = "/Users/finnsommer/llama-finetune/Data/Mark_Twain/Final"
# '''

# Charles Dickens
# Comment to use
# '''
cleaned_folder = "/Users/finnsommer/llama-finetune/Data/Charles_Dickens/Cleaned"
eval_folder = "/Users/finnsommer/llama-finetune/Data/Charles_Dickens/Evaluation_Data"
final_folder = "/Users/finnsommer/llama-finetune/Data/Charles_Dickens/Final"
# '''

# Create destination folders if they do not exist.
for folder in [eval_folder, final_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Get all cleaned text files from the cleaned folder.
file_list = glob.glob(os.path.join(cleaned_folder, "**", "*.txt"), recursive=True)

# Initialize global lists for modified texts and evaluation paragraphs.
all_modified_texts = []
all_eval_paragraphs = []

# Process each file.
for file_path in file_list:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read file {file_path}: {e}")
        continue

    # Split the text into paragraphs using double newlines; remove empties.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 4:
        print(f"[WARNING] Not enough paragraphs in file {file_path} to divide into four parts. Skipping file.")
        continue

    # Divide the paragraphs into four parts.
    num_paragraphs = len(paragraphs)
    part_size = num_paragraphs // 4
    part1 = paragraphs[:part_size]
    part2 = paragraphs[part_size:2*part_size]
    part3 = paragraphs[2*part_size:3*part_size]
    part4 = paragraphs[3*part_size:]
    
    # For each part, extract a sample paragraph (choose the middle) and remove it.
    eval_paragraphs = []
    new_parts = []
    for part in [part1, part2, part3, part4]:
        if not part:
            print(f"[WARNING] One of the parts in {file_path} is empty. Skipping extraction for that part.")
            continue
        mid_idx = len(part) // 2
        eval_paragraph = part[mid_idx]
        eval_paragraphs.append(eval_paragraph)
        # Remove the sample paragraph.
        new_part = part[:mid_idx] + part[mid_idx+1:]
        new_parts.append(new_part)
        
    # Append these evaluation paragraphs to the global evaluation list.
    all_eval_paragraphs.extend(eval_paragraphs)
    
    # Reconstruct the modified text (all paragraphs minus the extracted samples).
    modified_paragraphs = [p for part in new_parts for p in part]
    modified_text = "\n\n".join(modified_paragraphs)
    all_modified_texts.append(modified_text)
    
    # Delete the original file.
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"[WARNING] Could not delete original file {file_path}: {e}")

# After processing all files, save the evaluation samples and final concatenated text.

# Save all evaluation paragraphs into one big file.
eval_file = os.path.join(eval_folder, "all_evaluation_paragraphs.txt")
try:
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_eval_paragraphs))
except Exception as e:
    print(f"[ERROR] Could not write evaluation file {eval_file}: {e}")

# Concatenate all modified texts (with evaluation paragraphs removed) into one final file.
final_text = "\n\n".join(all_modified_texts)
final_file = os.path.join(final_folder, "final_concatenated.txt")
try:
    with open(final_file, "w", encoding="utf-8") as f:
        f.write(final_text)
except Exception as e:
    print(f"[ERROR] Could not write final concatenated file {final_file}: {e}")

print("Processing complete.")
