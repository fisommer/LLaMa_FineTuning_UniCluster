import pandas as pd

# List of CSV file paths (adjust as needed)
FILE_LIST = [
    "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Seed List /style_anno_1.csv",
    "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Seed List /style_anno_2.csv",
    "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Seed List /style_anno_3.csv",
    "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Seed List /style_anno_4.csv",
    "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Seed List /style_anno_5.csv",
]   

# The expected categories from the CSV files
CATEGORIES = ["literary", "abstract", "objective", "colloquial", "concrete", "subjective"]

def to_binary(val):
    """
    Convert an annotation cell to a binary indicator.
    Any non-empty, non-zero value counts as a vote.
    """
    if pd.isnull(val):
        return 0
    s = str(val).strip().lower()
    return 0 if s == "" or s == "0" else 1

def load_text_seeds(filepath):
    """
    Load seed words from a text file.
    Assumes one seed word per line.
    Returns a list of seed words.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            seeds = [line.strip() for line in f if line.strip()]
        return seeds
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

def load_seed_words():
    """
    Reads all seed CSV files and returns a dictionary with keys as categories and values
    as lists of seed words (for the CSV-based categories). Then, loads additional seed words
    for the categories 'formal' and 'informal' from text files.
    A word is assigned to a CSV-based category only if it gets a majority of votes
    from the annotator files.
    """
    dfs = []
    for filename in FILE_LIST:
        try:
            # Read CSV using semicolon as delimiter.
            df = pd.read_csv(filename, delimiter=";", skipinitialspace=True, engine="python")
            # Clean column names: strip whitespace and lower-case them.
            df.columns = [col.strip().lower() for col in df.columns if isinstance(col, str)]
            # Remove extra unnamed columns (if any)
            df = df.loc[:, ~df.columns.str.contains("^unnamed")]
            
            # If "word" is not present, assume the first column is the word column.
            if "word" not in df.columns:
                first_col = df.columns[0]
                df.rename(columns={first_col: "word"}, inplace=True)
                print(f"Warning: In file {filename}, renamed column '{first_col}' to 'word'.")
            
            # Remove rows where the word column literally equals "word" (repeated header rows)
            df = df[df["word"].str.lower() != "word"]
            
            # Keep only the expected columns if they exist.
            expected_cols = ["word"] + CATEGORIES
            df = df[[col for col in df.columns if col in expected_cols]]
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Skipping file.")
    
    if not dfs:
        raise ValueError("No valid seed CSV files could be read.")
    
    # Combine all annotator data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert annotation columns to binary votes.
    for cat in CATEGORIES:
        combined_df[cat] = combined_df[cat].apply(to_binary)
    
    # Group by word and sum votes from all files.
    grouped = combined_df.groupby("word")[CATEGORIES].sum().reset_index()
    
    n_annotators = len(dfs)
    threshold = (n_annotators // 2) + 1  # Majority threshold
    
    # Create seed lists dictionary for CSV-based categories.
    seed_lists = {cat: [] for cat in CATEGORIES}
    
    # For each word, assign it to the category with the maximum votes (if it meets threshold).
    for _, row in grouped.iterrows():
        word = row["word"]
        votes = {cat: row[cat] for cat in CATEGORIES}
        max_votes = max(votes.values())
        if max_votes >= threshold:
            winning_cats = [cat for cat, count in votes.items() if count == max_votes]
            chosen_cat = winning_cats[0]  # Tie-breaker: first category in the list.
            seed_lists[chosen_cat].append(word)
    
    # Now, load additional seed words for formal and informal categories from text files.
    formal_path = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Seed List /formal_seeds_100.txt"
    informal_path = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Seed List /informal_seeds_100.txt"
    
    seed_lists["formal"] = load_text_seeds(formal_path)
    seed_lists["informal"] = load_text_seeds(informal_path)
    
    return seed_lists

if __name__ == "__main__":
    seeds = load_seed_words()
    for cat in sorted(seeds.keys()):
        print(f"{cat.capitalize()} seeds ({len(seeds[cat])} words):")
        print(seeds[cat])
        print()
