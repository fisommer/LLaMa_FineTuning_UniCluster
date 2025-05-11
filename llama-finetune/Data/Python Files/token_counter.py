from transformers import PreTrainedTokenizerFast

def main():
    # point this at your HF model dir's tokenizer.json
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=(
            "/pfs/work9/workspace/"
            "scratch/ma_fisommer-Dataset/"
            "hf_model/tokenizer.json"
        )
    )

    eval_path = (
        "/pfs/work9/workspace/"
        "scratch/ma_fisommer-Dataset/"
        "llama-finetune/Data/"
        "Charles_Dickens/Evaluation_Data/"
        "all_evaluation_paragraphs.txt"
    )

    # read your Dickens evaluation text
    with open(eval_path, encoding="utf-8") as f:
        text = f.read()

    # count tokens
    token_ids = tokenizer.encode(text)
    print(f"Evaluation set is {len(token_ids)} tokens")

if __name__ == "__main__":
    main()
