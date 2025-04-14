#!/usr/bin/env python3
import os
import glob
import re
import shutil
import pandas as pd

# --------------------------
# SETTINGS and PATHS
# --------------------------

# Path to the CSV file
csv_file = '/Users/finnsommer/llama-finetune/Data/pg_catalog.csv'

# Folder with all the original text files from Gutenberg.
books_folder = '/Users/finnsommer/llama-finetune/Data/Books_txt_Files'

# Base data folder
base_data_folder = '/Users/finnsommer/llama-finetune/Data'

# --------------------------
# FOLDER STRUCTURE
# --------------------------
# For the chosen author, we will use a folder named accordingly inside base_data_folder.
# Under that folder, we will create:
#   - Raw: Files copied from Books_txt_Files per CSV filter.
#   - Cleaned: Files after removing Gutenberg header/footer.
#   - Final: Modified files after extracting evaluation paragraphs.
#   - Evaluation_Data: Concatenated evaluation paragraphs from each file.

def get_author_base(author):
    if author.lower() in ["mark twain", "twain, mark"]:
        return os.path.join(base_data_folder, "Mark_Twain")
    elif author.lower() in ["charles dickens", "dickens, charles"]:
        return os.path.join(base_data_folder, "Charles_Dickens")
    else:
        return None

def create_subfolders(author_base):
    subfolders = ["Raw", "Cleaned", "Final", "Evaluation_Data"]
    for folder in subfolders:
        path = os.path.join(author_base, folder)
        if not os.path.exists(path):
            os.makedirs(path)
    return {name: os.path.join(author_base, name) for name in subfolders}

# --------------------------
# STEP 1: Copy Files based on CSV Filter
# --------------------------
def copy_raw_files(csv_file, books_folder, raw_folder, author_query):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[ERROR] Could not read CSV file: {e}")
        return

    # Filter rows where the Authors column contains the given author string and language is "en"
    filtered_df = df[
        (df['Authors'].str.contains(author_query, na=False)) &
        (df['Language'].str.lower() == "en")
    ]
    if filtered_df.empty:
        print(f"[WARNING] No entries found for author {author_query} in English.")
        return

    # Loop over filtered rows and copy files to the raw folder.
    for index, row in filtered_df.iterrows():
        text_num = str(row['Text#']).strip()
        source_file = os.path.join(books_folder, text_num, f"pg{text_num}.txt")
        if os.path.exists(source_file):
            dest_file = os.path.join(raw_folder, f"pg{text_num}.txt")
            try:
                shutil.copy2(source_file, dest_file)
            except Exception as e:
                print(f"[ERROR] Copy failed for {source_file}: {e}")
        else:
            print(f"[WARNING] Source file not found: {source_file}")

# --------------------------
# STEP 2: Clean Raw Files (Remove Gutenberg Header/Footer)
# --------------------------
def clean_raw_files(raw_folder, cleaned_folder):
    # Define markers and regex for title extraction.
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker   = "*** END OF THE PROJECT GUTENBERG"
    start_pattern = re.compile(
        r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK\s*(.*?)\s*\*\*\*",
        re.IGNORECASE | re.DOTALL
    )

    # Find all raw txt files.
    file_list = glob.glob(os.path.join(raw_folder, "*.txt"))
    for file_path in file_list:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read file {file_path}: {e}")
            continue

        start_index = text.find(start_marker)
        if start_index == -1:
            print(f"[WARNING] Start marker not found in {file_path}. Skipping file.")
            continue

        match = start_pattern.search(text)
        if match:
            title = match.group(1).strip()
        else:
            print(f"[WARNING] Title extraction failed in {file_path}. Skipping file.")
            continue

        safe_title = re.sub(r"[^\w\s-]", "", title).strip()
        safe_title = re.sub(r"\s+", "_", safe_title)
        if not safe_title:
            print(f"[WARNING] Sanitized title empty in {file_path}. Skipping file.")
            continue

        header_line_end = text.find("\n", start_index)
        if header_line_end == -1:
            header_line_end = start_index + len(start_marker)
        else:
            header_line_end += 1  # remove the header line

        cleaned_text = text[header_line_end:]
        end_index = cleaned_text.find(end_marker)
        if end_index != -1:
            cleaned_text = cleaned_text[:end_index]
        else:
            print(f"[WARNING] End marker not found in {file_path}. Keeping text from header onward.")

        dest_file = os.path.join(cleaned_folder, f"{safe_title}.txt")
        try:
            with open(dest_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
        except Exception as e:
            print(f"[ERROR] Could not write cleaned file {dest_file}: {e}")
            continue

        # Remove the raw file.
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[WARNING] Could not delete raw file {file_path}: {e}")

# --------------------------
# STEP 3: Extract Evaluation Paragraphs and Produce Final Files
# --------------------------
def process_cleaned_files(cleaned_folder, final_folder, eval_folder):
    file_list = glob.glob(os.path.join(cleaned_folder, "*.txt"))
    all_modified_texts = []
    all_eval_paragraphs = []

    for file_path in file_list:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read file {file_path}: {e}")
            continue

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 4:
            print(f"[WARNING] Not enough paragraphs in {file_path} to divide. Skipping file.")
            continue

        num_paragraphs = len(paragraphs)
        part_size = num_paragraphs // 4
        part1 = paragraphs[:part_size]
        part2 = paragraphs[part_size:2*part_size]
        part3 = paragraphs[2*part_size:3*part_size]
        part4 = paragraphs[3*part_size:]
        
        eval_paragraphs = []
        new_parts = []
        for part in [part1, part2, part3, part4]:
            if not part:
                print(f"[WARNING] One part is empty in {file_path}. Skipping that part.")
                continue
            mid_idx = len(part) // 2
            eval_paragraph = part[mid_idx]
            eval_paragraphs.append(eval_paragraph)
            new_part = part[:mid_idx] + part[mid_idx+1:]
            new_parts.append(new_part)

        all_eval_paragraphs.extend(eval_paragraphs)
        modified_paragraphs = [p for part in new_parts for p in part]
        modified_text = "\n\n".join(modified_paragraphs)
        all_modified_texts.append(modified_text)

        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[WARNING] Could not delete cleaned file {file_path}: {e}")

    # Save all evaluation paragraphs into one evaluation file.
    eval_file = os.path.join(eval_folder, "all_evaluation_paragraphs.txt")
    try:
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_eval_paragraphs))
    except Exception as e:
        print(f"[ERROR] Could not write evaluation file {eval_file}: {e}")

    # Concatenate all modified texts into one final file.
    final_text = "\n\n".join(all_modified_texts)
    final_file = os.path.join(final_folder, "final_concatenated.txt")
    try:
        with open(final_file, "w", encoding="utf-8") as f:
            f.write(final_text)
    except Exception as e:
        print(f"[ERROR] Could not write final concatenated file {final_file}: {e}")

# --------------------------
# MAIN WORKFLOW
# --------------------------
def main():
    author_input = input("Enter author name (Mark Twain or Charles Dickens): ").strip()
    author_folder = get_author_base(author_input)
    if author_folder is None:
        print("[ERROR] Unsupported author provided.")
        return

    folders = create_subfolders(author_folder)
    raw_folder = folders["Raw"]
    cleaned_folder = folders["Cleaned"]
    final_folder = folders["Final"]
    eval_folder = folders["Evaluation_Data"]

    # Step 1: Copy raw files from Books_txt_Files using CSV filter.
    # For Mark Twain, use "Twain, Mark". For Charles Dickens, use "Dickens, Charles".
    if author_input.lower().startswith("mark"):
        author_query = "Twain, Mark"
    elif author_input.lower().startswith("charles"):
        author_query = "Dickens, Charles"
    else:
        print("[ERROR] Author query not recognized.")
        return

    copy_raw_files(csv_file, books_folder, raw_folder, author_query)

    # Step 2: Clean the raw files (remove Gutenberg header/footer, etc.)
    clean_raw_files(raw_folder, cleaned_folder)

    # Step 3: Process cleaned files to extract evaluation paragraphs and produce final texts.
    process_cleaned_files(cleaned_folder, final_folder, eval_folder)

    print("All processing complete.")

if __name__ == "__main__":
    main()
