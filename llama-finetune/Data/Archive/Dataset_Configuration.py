#!/usr/bin/env python3
import re
import sys

def process_tom_sawyer_style(input_path, output_path):
    """
    1) Reads all lines from the input file.
    2) Iterates through the file looking for "CHAPTER I" that is immediately followed
       (in the next three lines) by no other occurrence of the word "CHAPTER".
       - For each line checked, a checkpoint message is printed.
       - When a match is found, the script prints that line plus the next three lines.
       - If one of the next three lines contains "CHAPTER", the occurrence is skipped.
       - Otherwise, the script deletes everything before that line.
    3) Removes everything from (and including) any line containing '*** END OF'
       to the end of the file.
    4) Writes the resulting lines to output_path.
    """
    # Read entire file into a list of lines
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    # Compile the regex for "CHAPTER I"
    # It matches "CHAPTER I" when followed by end-of-line, a space, or a period.
    pattern = re.compile(r"(?i)\bCHAPTER\s+I\b(?=$|[ .])")

    valid_index = None
    # Iterate over lines to find a valid occurrence.
    for i, line in enumerate(lines):
        print(f"Checking line {i}: {line.strip()}")
        if pattern.search(line):
            print(f"Found 'CHAPTER I' at line {i}:")
            # Print this line and the next ten lines (if available)
            for j in range(i, min(i + 11, len(lines))):
                print(f"  Line {j}: {lines[j].strip()}")
            # Check the next ten lines for any occurrence of "CHAPTER"
            found_additional = False
            for k in range(i + 1, min(i + 11, len(lines))):
                if "CHAPTER" in lines[k].upper():
                    print(f"  Check: Found another 'CHAPTER' at line {k}: {lines[k].strip()} -- skipping this occurrence.")
                    found_additional = True
                    break
            if not found_additional:
                print(f"No additional 'CHAPTER' found in the next ten lines after line {i}.")
                valid_index = i
                break

    if valid_index is None:
        print("No valid 'CHAPTER I' occurrence found. No trimming performed on preamble.")
        trimmed_lines = lines
    else:
        # Delete everything before the valid occurrence.
        print(f"Deleting everything before line {valid_index} (the valid 'CHAPTER I' occurrence).")
        trimmed_lines = lines[valid_index:]

    # Remove everything from (and including) the line that contains '*** END OF'
    idx_end = None
    for i, line in enumerate(trimmed_lines):
        if "*** END OF" in line.upper():
            idx_end = i
            print(f"Found end marker '*** END OF' at line {i} in trimmed text.")
            break
    if idx_end is not None:
        trimmed_lines = trimmed_lines[:idx_end]

    # Write out the final lines
    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(trimmed_lines)
    print(f"Processed output saved to '{output_path}'.")

if __name__ == "__main__":
    
    # Process Tom Sawyer
    input_file = "/Users/finnsommer/llama-finetune/Data/Mark_Twain_TomSawyer.txt"
    output_file = "/Users/finnsommer/llama-finetune/Data/TomSawyer_processed.txt"
    process_tom_sawyer_style(input_file, output_file)
    print(f"Done processing '{input_file}' -> '{output_file}'.")

    # Process Huckleberry Finn
    input_file = "/Users/finnsommer/llama-finetune/Data/Mark_Twain_HuckleberryFinn#.txt"
    output_file = "/Users/finnsommer/llama-finetune/Data/Mark_Twain_HuckleberryFinn_processed.txt"
    process_tom_sawyer_style(input_file, output_file)
    print(f"Done processing '{input_file}' -> '{output_file}'.")

    # Process Charles Dickens' A Tale of Two Cities
    input_file = "/Users/finnsommer/llama-finetune/Data/Charles_Dickens_ATaleOfTwoCities.txt"
    output_file = "/Users/finnsommer/llama-finetune/Data/Charles_Dickens_ATaleOfTwoCities_processed.txt"
    process_tom_sawyer_style(input_file, output_file)
    print(f"Done processing '{input_file}' -> '{output_file}'.")
    
    # Process Charles Dickens' Great Expectations
    input_file = "/Users/finnsommer/llama-finetune/Data/Chalres_Dickens_Great_Expectations.txt"
    output_file = "/Users/finnsommer/llama-finetune/Data/Chalres_Dickens_Great_Expectations_processed.txt"
    process_tom_sawyer_style(input_file, output_file)
    print(f"Done processing '{input_file}' -> '{output_file}'.")



