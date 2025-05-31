import os
import random # Though not used for every Nth, good to have if logic changes
import logging
import re # For paragraph splitting logic, though not strictly needed for simple Nth

# --- Path Configuration ---
# The script itself will be at: /Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Python Files/simple_sampler.py
# We'll define base paths and then specify author-specific parts.
BASE_DATA_DIR = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data"

def sample_every_nth_paragraph(input_filepath, output_filepath, log_filepath, n=4, author_name="UnknownAuthor"):
    """
    Creates a smaller sample from a text file by taking every Nth paragraph.
    Assumes paragraphs are separated by at least one blank line.

    Args:
        input_filepath (str): Path to the input text file (e.g., your existing eval.txt).
        output_filepath (str): Path where the sampled output will be saved.
        log_filepath (str): Path to the log file.
        n (int): The sampling interval (e.g., 4 to take every 4th paragraph).
        author_name (str): Name of the author for logging/output purposes.
    """
    # --- Setup Logging ---
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    logger = logging.getLogger(f"simple_sampler_{author_name}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any to avoid duplicate logging in interactive sessions
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_filepath, mode='w', encoding='utf-8') # 'w' to overwrite log each run
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"📖 Starting 'every {n}th paragraph' sampling for {author_name}")
    logger.info(f"Input file: {input_filepath}")
    logger.info(f"Output file: {output_filepath}")
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Sampling interval (N): {n}")


    sampled_paragraphs = []
    paragraph_buffer = []
    paragraph_count = 0 # Counts actual non-empty paragraphs processed
    original_lines_count = 0 # For basic file size indication

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line_number, line_content in enumerate(f, 1):
                original_lines_count += 1
                stripped_line = line_content.strip()
                if stripped_line:  # Non-empty line, part of a paragraph
                    paragraph_buffer.append(line_content)
                elif paragraph_buffer:  # Empty line signifies end of a paragraph
                    paragraph_text = "".join(paragraph_buffer).strip()
                    if paragraph_text: # Ensure we don't count paragraphs made of only whitespace
                        paragraph_count += 1
                        # For 1-based Nth (e.g., 1st, 4th, 8th if N=4 and we start counting paragraphs from 1)
                        # If N=4, we want paragraphs 4, 8, 12...
                        # So, if paragraph_count is a multiple of N
                        if paragraph_count % n == 0:
                            sampled_paragraphs.append(paragraph_text)
                            logger.info(f"Sampled paragraph #{paragraph_count} (overall) from input line ~{line_number - len(paragraph_buffer)}")
                    paragraph_buffer = [] 
            
            # Add the last paragraph if the file doesn't end with a blank line
            if paragraph_buffer:
                paragraph_text = "".join(paragraph_buffer).strip()
                if paragraph_text:
                    paragraph_count += 1
                    if paragraph_count % n == 0:
                        sampled_paragraphs.append(paragraph_text)
                        logger.info(f"Sampled paragraph #{paragraph_count} (overall) from input line ~{original_lines_count - len(paragraph_buffer) +1 } (last paragraph)")
        
        logger.info(f"📊 Total original lines processed: {original_lines_count}")
        logger.info(f"📊 Total non-empty paragraphs found and processed in original file: {paragraph_count}")

    except FileNotFoundError:
        logger.error(f"❌ Error: Input file not found at {input_filepath}")
        print(f"❌ Error: Input file not found at {input_filepath}")
        return
    except Exception as e:
        logger.error(f"❌ An error occurred while reading the file: {e}")
        print(f"❌ An error occurred while reading the file: {e}")
        return

    if not sampled_paragraphs:
        message = "⚠️ No paragraphs were sampled. Output will be empty. Check sampling interval 'N' or input file content/structure."
        logger.warning(message)
        print(message)
        # Create an empty output file
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write("")
            logger.info(f"Created empty output file at: {output_filepath}")
        except Exception as e:
            logger.error(f"❌ An error occurred while writing the empty output file: {e}")
            print(f"❌ An error occurred while writing the empty output file: {e}")
        return

    # --- Save the sampled paragraphs ---
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for i, para_text in enumerate(sampled_paragraphs):
                f.write(para_text)
                if i < len(sampled_paragraphs) - 1:
                    f.write("\n\n")  # Ensure paragraphs are separated by a blank line
        
        final_message = f"✅ Successfully created sampled file with {len(sampled_paragraphs)} paragraphs (every {n}th) at: {output_filepath}"
        logger.info(final_message)
        print(f"\n{final_message}")
    except Exception as e:
        logger.error(f"❌ An error occurred while writing the output file: {e}")
        print(f"❌ An error occurred while writing the output file: {e}")

# --- Main execution ---
if __name__ == "__main__":
    print("-----------------------------------------------------")
    print("      Evaluation File Paragraph Sampler (Every Nth)  ")
    print("-----------------------------------------------------")

    author_choice_input = ""
    while author_choice_input not in ['d', 't']:
        author_choice_input = input("Which author's eval.txt file do you want to shrink?\n[d]ickens or [t]wain: ").strip().lower()
        if author_choice_input.startswith('d'):
            author_key = "Charles_Dickens"
            author_name_proper = "Charles Dickens"
        elif author_choice_input.startswith('t'):
            author_key = "Mark_Twain"
            author_name_proper = "Mark Twain"
        else:
            print("Invalid choice. Please enter 'd' for Dickens or 't' for Twain.")
            author_choice_input = "" # Reset to loop

    # Construct paths based on author choice
    input_eval_file = os.path.join(BASE_DATA_DIR, author_key, "Splits", "eval.txt")
    log_dir = os.path.join(BASE_DATA_DIR, author_key, "Log Files")
    
    sampling_interval_n = 0
    while sampling_interval_n <= 0:
        try:
            interval_str = input(f"Sample every Nth paragraph for {author_name_proper} (e.g., enter 4 to take every 4th): ").strip()
            sampling_interval_n = int(interval_str)
            if sampling_interval_n <= 0:
                print("Please enter a positive integer for N.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
            sampling_interval_n = 0 # Ensure loop continues

    # Define output and log file names
    output_sampled_file = os.path.join(BASE_DATA_DIR, author_key, "Splits", f"eval_every_{sampling_interval_n}th_para_sample.txt")
    log_file_path = os.path.join(log_dir, f"simple_sampler_eval_every_{sampling_interval_n}th.log")
    
    # Check if input file exists before proceeding
    if not os.path.exists(input_eval_file):
        print(f"❌ Error: Input file '{input_eval_file}' not found.")
        print("Please ensure the file exists at the specified path and the BASE_DATA_DIR is correct.")
    else:
        sample_every_nth_paragraph(
            input_filepath=input_eval_file,
            output_filepath=output_sampled_file,
            log_filepath=log_file_path,
            n=sampling_interval_n,
            author_name=author_name_proper
        )
        print(f"\n🔍 You can inspect the output file at: {output_sampled_file}")
        print(f"🪵  Log file has been saved to: {log_file_path}")