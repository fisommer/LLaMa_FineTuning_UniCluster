#!/usr/bin/env python3
import os
import re
import logging

# --- Configuration ---
BASE_DATA_DIR = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data"
DEFAULT_NUM_PROMPTS_TO_EXTRACT = 21
TARGET_PROMPT_LENGTH_WORDS = 30  # The desired length for prompts
MIN_INITIAL_PARA_WORDS = 10       # A paragraph must have at least this many words to start a prompt
MIN_FINAL_PROMPT_WORDS = 20      # A final prompt must have at least this many words

# Regex to identify chapter lines (case-insensitive)
CHAPTER_PATTERN = re.compile(
    r"^\s*"
    r"(CHAPTER|CHPT\.?|CH\.?|SECTION|Part|PART|Book|BOOK|ACT|Act|SCENE|Scene|STAVE)"
    r"\s+[IVXLCDM\d\w\s\-\—\.'’:]*"
    r"\s*$",
    re.IGNORECASE
)

# Logger will be configured in main()
logger = logging.getLogger(__name__)

def setup_logging(author_key_for_log):
    global logger
    log_dir_author_specific = os.path.join(BASE_DATA_DIR, author_key_for_log, "Log Files")
    os.makedirs(log_dir_author_specific, exist_ok=True)
    log_file_path_author_specific = os.path.join(log_dir_author_specific, "prompt_preparation_v2.log")

    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(formatter); logger.addHandler(ch)
    try:
        fh = logging.FileHandler(log_file_path_author_specific, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO); fh.setFormatter(formatter); logger.addHandler(fh)
        print(f"Logging to: {log_file_path_author_specific}")
    except Exception as e:
        print(f"Error setting up file logger at {log_file_path_author_specific}: {e}")

def extract_and_save_prompts(input_eval_filepath, output_prompts_filepath,
                             num_prompts_to_extract=DEFAULT_NUM_PROMPTS_TO_EXTRACT,
                             target_prompt_length=TARGET_PROMPT_LENGTH_WORDS,
                             min_initial_len=MIN_INITIAL_PARA_WORDS,
                             min_final_len=MIN_FINAL_PROMPT_WORDS,
                             author_name="UnknownAuthor"):
    logger.info(f"Starting prompt extraction for {author_name} from: {input_eval_filepath}")
    logger.info(f"Targeting {num_prompts_to_extract} prompts, each approx. {target_prompt_length} words.")
    logger.info(f"Prompts will be saved to: {output_prompts_filepath}")

    extracted_prompts = [] # This is the correct list name

    try:
        with open(input_eval_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"Input evaluation file not found: {input_eval_filepath}"); return False
    except Exception as e:
        logger.error(f"Error reading {input_eval_filepath}: {e}", exc_info=True); return False

    all_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
    if not all_paragraphs:
        logger.warning(f"No paragraphs found in {input_eval_filepath}."); return False
    logger.info(f"Found {len(all_paragraphs)} non-empty paragraphs in the input file.")

    current_para_idx = 0
    while len(extracted_prompts) < num_prompts_to_extract and current_para_idx < len(all_paragraphs):
        base_paragraph_text = all_paragraphs[current_para_idx]

        if CHAPTER_PATTERN.match(base_paragraph_text.strip()): # Use CHAPTER_PATTERN here
            logger.debug(f"Index {current_para_idx}: Skipped chapter line: '{base_paragraph_text[:70]}...'")
            current_para_idx += 1
            continue

        words_from_base_para = base_paragraph_text.split()
        if len(words_from_base_para) < min_initial_len:
            logger.debug(f"Index {current_para_idx}: Base paragraph too short ({len(words_from_base_para)} words). Min required: {min_initial_len}. Content: '{base_paragraph_text[:70]}...'")
            current_para_idx += 1
            continue

        current_prompt_word_list = list(words_from_base_para[:target_prompt_length])

        if len(current_prompt_word_list) < target_prompt_length and \
           len(words_from_base_para) < target_prompt_length:
            logger.debug(f"Index {current_para_idx}: Prompt from base is short ({len(current_prompt_word_list)} words from original {len(words_from_base_para)}). Attempting to append...")
            idx_for_append = current_para_idx + 1
            while len(current_prompt_word_list) < target_prompt_length and idx_for_append < len(all_paragraphs):
                next_para_to_append_text = all_paragraphs[idx_for_append]
                if CHAPTER_PATTERN.match(next_para_to_append_text.strip()): # Use CHAPTER_PATTERN
                    logger.debug(f"  Append skip: Index {idx_for_append} is chapter line: '{next_para_to_append_text[:70]}...'")
                    idx_for_append += 1
                    continue
                words_from_next_para = next_para_to_append_text.split()
                if not words_from_next_para:
                    idx_for_append += 1; continue
                words_needed_for_prompt = target_prompt_length - len(current_prompt_word_list)
                words_to_take_from_next = words_from_next_para[:words_needed_for_prompt]
                current_prompt_word_list.extend(words_to_take_from_next)
                logger.debug(f"  Appended {len(words_to_take_from_next)} words from para {idx_for_append}. Prompt now {len(current_prompt_word_list)} words.")
                if len(words_from_next_para) >= words_needed_for_prompt: break
                idx_for_append += 1

        final_prompt_text = " ".join(current_prompt_word_list)
        final_prompt_word_count = len(final_prompt_text.split())

        if final_prompt_word_count >= min_final_len:
            if final_prompt_text not in extracted_prompts:
                extracted_prompts.append(final_prompt_text)
                logger.info(f"Added prompt #{len(extracted_prompts)} (from para {current_para_idx}, {final_prompt_word_count} words): '{final_prompt_text[:70]}...'")
            else:
                logger.debug(f"Skipping duplicate fully formed prompt: '{final_prompt_text[:70]}...'")
        else:
            logger.debug(f"Final prompt from para {current_para_idx} too short ({final_prompt_word_count} words). Min required: {min_final_len}. Skipping: '{final_prompt_text[:70]}...'")
        current_para_idx += 1

    if len(extracted_prompts) < num_prompts_to_extract:
        logger.warning(f"Only extracted {len(extracted_prompts)} unique prompts, less than requested {num_prompts_to_extract}.")
    if not extracted_prompts:
        logger.warning(f"No prompts extracted for {author_name}. Output file will not be created."); return False

    output_dir = os.path.dirname(output_prompts_filepath)
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except Exception as e:
        logger.error(f"Could not create output dir {output_dir}: {e}", exc_info=True); return False

    try:
        with open(output_prompts_filepath, 'w', encoding='utf-8') as f:
            for i, prompt_text_to_write in enumerate(extracted_prompts): # Changed variable name for clarity
                f.write(prompt_text_to_write)
                # Use the correct list name here: extracted_prompts
                if i < len(extracted_prompts) - 1: 
                    f.write("\n")
        logger.info(f"Successfully saved {len(extracted_prompts)} prompts to {output_prompts_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error writing prompts to {output_prompts_filepath}: {e}", exc_info=True); return False

def main():
    author_key, author_name_proper = "", ""
    while True:
        choice = input("Which author's eval file for prompts? [d]ickens or [t]wain: ").strip().lower()
        if choice.startswith('d'): author_key, author_name_proper = "Charles_Dickens", "Charles Dickens"; break
        elif choice.startswith('t'): author_key, author_name_proper = "Mark_Twain", "Mark Twain"; break
        else: print("Invalid choice.")

    setup_logging(author_key) # Setup logging after author is known
    logger.info("--- Starting Prompt Preparation Script (Improved Logic) ---")
    logger.info(f"Selected author: {author_name_proper}")

    eval_sample_fn = "eval_every_15th_para_sample.txt"
    input_eval_f = os.path.join(BASE_DATA_DIR, author_key, "Splits", eval_sample_fn)
    prompts_out_dir = os.path.join(BASE_DATA_DIR, author_key, "Prompts")
    # Changed filename slightly to reflect it's from v2 of this script logic
    prompts_out_fn = f"{author_key.lower().replace('_', '')}_prompts_v2_{DEFAULT_NUM_PROMPTS_TO_EXTRACT}.txt"
    output_prompts_f = os.path.join(prompts_out_dir, prompts_out_fn)

    if not os.path.exists(input_eval_f):
        logger.error(f"CRITICAL: Input file not found: {input_eval_f}"); print(f"Error: Input file missing."); return

    extract_and_save_prompts(
        input_eval_filepath=input_eval_f, output_prompts_filepath=output_prompts_f,
        num_prompts_to_extract=DEFAULT_NUM_PROMPTS_TO_EXTRACT,
        target_prompt_length=TARGET_PROMPT_LENGTH_WORDS,
        min_initial_len=MIN_INITIAL_PARA_WORDS,
        min_final_len=MIN_FINAL_PROMPT_WORDS,
        author_name=author_name_proper
    )
    logger.info("--- Prompt Preparation Script Finished ---")

if __name__ == "__main__":
    main()
