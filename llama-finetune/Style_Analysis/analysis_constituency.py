#!/usr/bin/env python3
"""
analysis_constituency.py

Revised script that uses Algorithm 1 from the paper to classify sentences into:
simple, compound, complex, complex-compound, and other, based on their constituency parse trees.
It also includes two lexical analyses: (a) a simple seed-count method and (b) a full lexical style
analysis using normalized PMI with seed word lists for eight categories, which are then combined into
four spectrums: subjective-objective, concrete-abstract, literary-colloquial, and formal-informal.

After launching, you will be prompted to choose which analyses to run.
"""
import math
import nltk
import re
import spacy
import benepar
from nltk import Tree
from scipy.spatial.distance import jensenshannon

# Import the seed loader module.
from seed_loader import load_seed_words

# Ensure that spaCy uses Benepar for constituency parsing.
nlp = spacy.load("en_core_web_sm")
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# Default file paths 
FILE_1 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/generated_55189.txt" 
FILE_2 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Charles_Dickens/Evaluation_Data/all_evaluation_paragraphs.txt" 
'''
# Define maximum expected differences for normalization
max_diffs = {
    "avg_words_per_sentence": 10,        # maximum expected difference in words per sentence
    "avg_commas_per_sentence": 1,          # maximum expected difference for commas
    "avg_semicolons_per_sentence": 0.3,       # maximum expected difference for semicolons
    "avg_colons_per_sentence": 0.1,           # maximum expected difference for colons
    "avg_sentences_per_paragraph": 1       # maximum expected difference for sentences per paragraph
}

'''
# ---------------------------
# Load Seed Words from CSV Files
# ---------------------------
seed_lists = load_seed_words()

# Load seeds and update global seed sets:
LITERARY_SEEDS = set(seed_lists.get("literary", []))
ABSTRACT_SEEDS = set(seed_lists.get("abstract", []))
OBJECTIVE_SEEDS = set(seed_lists.get("objective", []))
COLLOQUIAL_SEEDS = set(seed_lists.get("colloquial", []))
CONCRETE_SEEDS = set(seed_lists.get("concrete", []))
SUBJECTIVE_SEEDS = set(seed_lists.get("subjective", []))
FORMAL_SEEDS = set(seed_lists.get("formal", []))
INFORMAL_SEEDS = set(seed_lists.get("informal", []))


# ---------------------------
# Part 1: Surface-Level Features
# ---------------------------
def get_surface_features(text):
    print("[CHECKPOINT] Entering get_surface_features")
    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    print(f"[CHECKPOINT] Number of sentences: {num_sentences}")
    
    if num_sentences == 0:
        print("[CHECKPOINT] No sentences found, returning zeros.")
        return {
            "avg_words_per_sentence": 0.0,
            "avg_commas_per_sentence": 0.0,
            "avg_semicolons_per_sentence": 0.0,
            "avg_colons_per_sentence": 0.0,
            "avg_sentences_per_paragraph": 0.0
        }
    
    total_words = sum(len(nltk.word_tokenize(sent)) for sent in sentences)
    avg_words_per_sentence = total_words / num_sentences

    total_commas = sum(sent.count(",") for sent in sentences)
    total_semicolons = sum(sent.count(";") for sent in sentences)
    total_colons = sum(sent.count(":") for sent in sentences)

    avg_commas_per_sentence = total_commas / num_sentences
    avg_semicolons_per_sentence = total_semicolons / num_sentences
    avg_colons_per_sentence = total_colons / num_sentences

    paragraphs = text.split("\n\n")
    num_paragraphs = len(paragraphs)
    total_sents_in_paras = sum(len(nltk.sent_tokenize(p.strip())) for p in paragraphs if p.strip())
    avg_sentences_per_paragraph = total_sents_in_paras / num_paragraphs if num_paragraphs > 0 else 0.0

    return {
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_commas_per_sentence": avg_commas_per_sentence,
        "avg_semicolons_per_sentence": avg_semicolons_per_sentence,
        "avg_colons_per_sentence": avg_colons_per_sentence,
        "avg_sentences_per_paragraph": avg_sentences_per_paragraph
    }

def compare_surface_features(features1, features2):
    print("[CHECKPOINT] Comparing surface features with MSE")
    keys = [
        "avg_words_per_sentence",
        "avg_commas_per_sentence",
        "avg_semicolons_per_sentence",
        "avg_colons_per_sentence",
        "avg_sentences_per_paragraph"
    ]
    squared_sum = 0.0
    for k in keys:
        diff = (features1[k] - features2[k]) ** 2
        squared_sum += diff
        print(f"[CHECKPOINT] {k}: F1={features1[k]:.3f}, F2={features2[k]:.3f}, (diff^2)={diff:.6f}")
    mse = squared_sum / len(keys)
    similarity = 1 / (1 + mse)
    print(f"[CHECKPOINT] MSE for surface features = {mse:.6f}")
    print(f"[CHECKPOINT] Surface-Level Similarity (1/(1+MSE)) = {similarity:.4f}")
    return mse, similarity

'''
def compare_surface_features_normalized(features1, features2, max_diffs):
    """
    Compute a normalized Mean Squared Error (MSE) between two sets of surface features.
    Each feature difference is normalized by a predetermined maximum expected difference.
    """
    keys = [
        "avg_words_per_sentence",
        "avg_commas_per_sentence",
        "avg_semicolons_per_sentence",
        "avg_colons_per_sentence",
        "avg_sentences_per_paragraph"
    ]
    normalized_sum = 0.0
    for k in keys:
        # Normalize the absolute difference using the expected maximum difference for the feature.
        norm_diff = abs(features1[k] - features2[k]) / max_diffs.get(k, 1)
        normalized_sum += norm_diff ** 2
        print(f"[CHECKPOINT] {k}: F1={features1[k]:.3f}, F2={features2[k]:.3f}, normalized diff^2={norm_diff**2:.6f}")
    mse_norm = normalized_sum / len(keys)
    # The normalized MSE is now in the range [0, 1].
    similarity = 1 / (1 + mse_norm)
    print(f"[CHECKPOINT] Normalized MSE = {mse_norm:.6f}")
    print(f"[CHECKPOINT] Surface-Level Similarity (1/(1+normalized MSE)) = {similarity:.4f}")
    return mse_norm, similarity
'''
# ---------------------------
# Part 2: Syntactic-Level Features (using Algorithm 1)
# ---------------------------
def clean_text_for_parsing(text):
    print("[CHECKPOINT] Starting text cleaning.")
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'["“”]', '', text)
    text = re.sub(r'[-–—]', '', text)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,;:.!?])', r'\1', text)
    cleaned = text.strip()
    print(f"[CHECKPOINT] Finished text cleaning. Cleaned text length: {len(cleaned)} characters.")
    return cleaned

def classify_sentence_benepar_safe(sentence):
    sentence_clean = sentence.strip().replace("“", "\"").replace("”", "\"")
    if not sentence_clean:
        return "other"
    try:
        doc = nlp(sentence_clean)
        sent_span = list(doc.sents)[0]
        tree_str = sent_span._.parse_string
        if not tree_str:
            return "other"
        tree = Tree.fromstring(tree_str)
    except Exception:
        return "other"

    Ltop = [child.label() for child in tree if isinstance(child, Tree)]
    all_labels = [subtree.label() for subtree in tree.subtrees() if isinstance(subtree, Tree)]
    
    if "S" in Ltop:
        if "SBAR" not in all_labels:
            return "compound"
        else:
            return "complex-compound"
    elif "VP" in Ltop:
        if "SBAR" not in all_labels:
            return "simple"
        else:
            return "complex"
    else:
        return "other"

def get_syntactic_distribution(text):
    print("[CHECKPOINT] Entering get_syntactic_distribution")
    sentences = nltk.sent_tokenize(text)
    print(f"[CHECKPOINT] Found {len(sentences)} sentences for syntactic analysis")
    
    counts = {"simple": 0, "compound": 0, "complex": 0, "complex-compound": 0, "other": 0}
    failure_count = 0
    failed_sentences = []
    
    for sent in sentences:
        cat = classify_sentence_benepar_safe(sent)
        if cat == "other":
            failure_count += 1
            if len(failed_sentences) < 10:
                failed_sentences.append(sent)
        counts[cat] += 1

    total_sents = len(sentences)
    if total_sents == 0:
        print("[CHECKPOINT] No sentences - returning zero distribution.")
        return [0, 0, 0, 0, 0]
    
    print(f"[CHECKPOINT] {failure_count} sentences could not be parsed reliably.")
    if failed_sentences:
        print("[CHECKPOINT] Sample of up to 10 sentences that could not be parsed reliably:")
        for i, fs in enumerate(failed_sentences, start=1):
            print(f"  {i}. {fs}")
    
    dist = [
        counts["simple"] / total_sents,
        counts["compound"] / total_sents,
        counts["complex"] / total_sents,
        counts["complex-compound"] / total_sents,
        counts["other"] / total_sents
    ]
    print("[CHECKPOINT] Syntactic distribution:", dist)
    return dist

def compute_syntactic_jsd(dist1, dist2):
    print("[CHECKPOINT] Computing JSD for syntactic distribution")
    return jensenshannon(dist1, dist2)

# ---------------------------
# Part 3: Lexical Analysis - Simple Seed Count
# ---------------------------
def get_lexical_style_simple(text):
    print("[CHECKPOINT] Starting simple lexical analysis.")
    tokens = nltk.word_tokenize(text.lower())
    
    count_subj = sum(1 for token in tokens if token in SUBJECTIVE_SEEDS)
    count_obj  = sum(1 for token in tokens if token in OBJECTIVE_SEEDS)
    count_conc = sum(1 for token in tokens if token in CONCRETE_SEEDS)
    count_abst = sum(1 for token in tokens if token in ABSTRACT_SEEDS)
    count_lit  = sum(1 for token in tokens if token in LITERARY_SEEDS)
    count_coll = sum(1 for token in tokens if token in COLLOQUIAL_SEEDS)
    count_form = sum(1 for token in tokens if token in FORMAL_SEEDS)
    count_infm = sum(1 for token in tokens if token in INFORMAL_SEEDS)
    
    eps = 1e-6
    subj_obj_score = count_subj / (count_subj + count_obj + eps)
    conc_abst_score = count_conc / (count_conc + count_abst + eps)
    lit_coll_score  = count_lit / (count_lit + count_coll + eps)
    form_infm_score = count_form / (count_form + count_infm + eps)
    
    lexical_vector = {
        "subjective_obj": subj_obj_score,
        "concrete_abstract": conc_abst_score,
        "literary_colloquial": lit_coll_score,
        "formal_informal": form_infm_score
    }
    
    print("[CHECKPOINT] Simple lexical style vector (values in [0,1]):")
    for k, v in lexical_vector.items():
        print(f"  {k}: {v:.3f}")
    
    return lexical_vector

# ---------------------------
# Part 4: Lexical Analysis - Full Analysis using Normalized PMI
# ---------------------------
def get_lexical_style_full(text):
    print("[CHECKPOINT] Starting full lexical analysis using NPMI.")
    eps = 1e-6
    sentences = nltk.sent_tokenize(text)
    total_docs = len(sentences)
    if total_docs == 0:
        return {"subjective_obj": 0, "concrete_abstract": 0, "literary_colloquial": 0, "formal_informal": 0}
    
    docs = [set(nltk.word_tokenize(sent.lower())) for sent in sentences]
    vocabulary = set()
    for doc in docs:
        vocabulary.update(doc)
    
    df = {w: sum(1 for doc in docs if w in doc) for w in vocabulary}
    
    subj_seeds = set(w.lower() for w in SUBJECTIVE_SEEDS)
    obj_seeds  = set(w.lower() for w in OBJECTIVE_SEEDS)
    conc_seeds = set(w.lower() for w in CONCRETE_SEEDS)
    abst_seeds = set(w.lower() for w in ABSTRACT_SEEDS)
    lit_seeds  = set(w.lower() for w in LITERARY_SEEDS)
    coll_seeds = set(w.lower() for w in COLLOQUIAL_SEEDS)
    form_seeds = set(w.lower() for w in FORMAL_SEEDS)
    infm_seeds = set(w.lower() for w in INFORMAL_SEEDS)
    
    seed_p = {}
    all_seed_words = subj_seeds.union(obj_seeds, conc_seeds, abst_seeds, lit_seeds, coll_seeds, form_seeds, infm_seeds)
    for s in all_seed_words:
        seed_p[s] = df.get(s, 0) / total_docs

    def npmi(p_xy, p_x, p_y):
        if p_xy <= 0:
            return -1.0
        return (math.log(p_xy / (p_x * p_y) + eps)) / (-math.log(p_xy + eps))
    
    raw_scores = {w: {"subj": 0.0, "obj": 0.0, "conc": 0.0, "abst": 0.0, "lit": 0.0, "coll": 0.0, "form": 0.0, "infm": 0.0}
                  for w in vocabulary}
    
    for w in vocabulary:
        p_w = df[w] / total_docs
        for s in subj_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["subj"] += npmi(p_ws, p_w, seed_p.get(s, 0))
        for s in obj_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["obj"] += npmi(p_ws, p_w, seed_p.get(s, 0))
        for s in conc_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["conc"] += npmi(p_ws, p_w, seed_p.get(s, 0))
        for s in abst_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["abst"] += npmi(p_ws, p_w, seed_p.get(s, 0))
        for s in lit_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["lit"] += npmi(p_ws, p_w, seed_p.get(s, 0))
        for s in coll_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["coll"] += npmi(p_ws, p_w, seed_p.get(s, 0))
        for s in form_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["form"] += npmi(p_ws, p_w, seed_p.get(s, 0))
        for s in infm_seeds:
            co = sum(1 for doc in docs if (w in doc and s in doc))
            p_ws = co / total_docs
            raw_scores[w]["infm"] += npmi(p_ws, p_w, seed_p.get(s, 0))
    
    def add_constant(score, seed_set):
        return score + len(seed_set)
    
    for w in vocabulary:
        raw_scores[w]["subj"] = add_constant(raw_scores[w]["subj"], subj_seeds)
        raw_scores[w]["obj"]  = add_constant(raw_scores[w]["obj"], obj_seeds)
        raw_scores[w]["conc"] = add_constant(raw_scores[w]["conc"], conc_seeds)
        raw_scores[w]["abst"] = add_constant(raw_scores[w]["abst"], abst_seeds)
        raw_scores[w]["lit"]  = add_constant(raw_scores[w]["lit"], lit_seeds)
        raw_scores[w]["coll"] = add_constant(raw_scores[w]["coll"], coll_seeds)
        raw_scores[w]["form"] = add_constant(raw_scores[w]["form"], form_seeds)
        raw_scores[w]["infm"] = add_constant(raw_scores[w]["infm"], infm_seeds)
    
    style_vectors = {}
    for w in vocabulary:
        subj_obj = raw_scores[w]["subj"] / (raw_scores[w]["subj"] + raw_scores[w]["obj"] + eps)
        conc_abst = raw_scores[w]["conc"] / (raw_scores[w]["conc"] + raw_scores[w]["abst"] + eps)
        lit_coll  = raw_scores[w]["lit"] / (raw_scores[w]["lit"] + raw_scores[w]["coll"] + eps)
        form_infm = raw_scores[w]["form"] / (raw_scores[w]["form"] + raw_scores[w]["infm"] + eps)
        style_vectors[w] = (subj_obj, conc_abst, lit_coll, form_infm)
    
    avg_style = [0.0, 0.0, 0.0, 0.0]
    for vec in style_vectors.values():
        avg_style[0] += vec[0]
        avg_style[1] += vec[1]
        avg_style[2] += vec[2]
        avg_style[3] += vec[3]
    n_words = len(style_vectors)
    avg_style = [x / n_words for x in avg_style]
    final_vector = {
        "subjective_obj": avg_style[0],
        "concrete_abstract": avg_style[1],
        "literary_colloquial": avg_style[2],
        "formal_informal": avg_style[3]
    }
    print("[CHECKPOINT] Full lexical style analysis vector (values in [0,1]):")
    for k, v in final_vector.items():
        print(f"  {k}: {v:.3f}")
    return final_vector

# ---------------------------
# Part 4.5: Lexical Analysis - Compare Lexical Vectors using MSE
# ---------------------------
def compare_lexical_vectors(vec1, vec2):
    print("[CHECKPOINT] Comparing lexical style vectors with MSE")
    keys = vec1.keys()  # assuming both dictionaries have the same keys
    squared_sum = 0.0
    for k in keys:
        diff = (vec1[k] - vec2[k]) ** 2
        squared_sum += diff
        print(f"[CHECKPOINT] {k}: V1={vec1[k]:.3f}, V2={vec2[k]:.3f}, (diff^2)={diff:.6f}")
    mse = squared_sum / len(keys)
    similarity = 1 / (1 + mse)
    print(f"[CHECKPOINT] MSE for lexical style vectors = {mse:.6f}")
    print(f"[CHECKPOINT] Lexical Style Similarity (1/(1+MSE)) = {similarity:.4f}")
    return mse, similarity

# ---------------------------
# Part 5: Main Program
# ---------------------------
def main():
    print(f"[CHECKPOINT] Reading File 1 from: {FILE_1}")
    with open(FILE_1, "r", encoding="utf-8") as f1:
        text1 = f1.read()
    print(f"[CHECKPOINT] Reading File 2 from: {FILE_2}")
    with open(FILE_2, "r", encoding="utf-8") as f2:
        text2 = f2.read()

    answer_surface = input("Do you want to perform surface-level analysis? [y|n]: ").strip().lower()
    if answer_surface in ["y", "yes"]:
        print("\n[CHECKPOINT] Performing surface-level analysis")
        surface1 = get_surface_features(text1)
        surface2 = get_surface_features(text2)
    
        print("\nSurface-Level Features for File 1:")
        for k, v in surface1.items():
            print(f"  {k}: {v:.3f}")
        print("\nSurface-Level Features for File 2:")
        for k, v in surface2.items():
            print(f"  {k}: {v:.3f}")
    
        #mse_norm, surface_similarity = compare_surface_features_normalized(surface1, surface2, max_diffs)
        #print(f"\nSurface-Level MSE: {mse_norm:.6f}")
        mse, surface_similarity = compare_surface_features(surface1, surface2)
        print(f"\nSurface-Level MSE: {mse:.6f}")
        print(f"Surface-Level Similarity (naive): {surface_similarity:.4f}")
    else:
        print("[CHECKPOINT] Skipping surface-level analysis.")

    answer_syntactic = input("\nDo you want to perform syntactic-level analysis? [y|n]: ").strip().lower()
    if answer_syntactic in ["y", "yes"]:
        print("\n[CHECKPOINT] Performing syntactic-level analysis")
        cleaned_text1 = clean_text_for_parsing(text1)
        cleaned_text2 = clean_text_for_parsing(text2)
    
        dist1 = get_syntactic_distribution(cleaned_text1)
        dist2 = get_syntactic_distribution(cleaned_text2)
    
        print("\nSyntactic Distribution (File 1):")
        print("  [Simple, Compound, Complex, Complex-Compound, Other]:")
        print("  ", [f"{p:.3f}" for p in dist1])
        print("Syntactic Distribution (File 2):")
        print("  [Simple, Compound, Complex, Complex-Compound, Other]:")
        print("  ", [f"{p:.3f}" for p in dist2])
    
        syntactic_jsd = compute_syntactic_jsd(dist1, dist2)
        print(f"\nSyntactic JSD (lower is more similar): {syntactic_jsd:.4f}")
        syntactic_similarity = 1.0 - syntactic_jsd
        print(f"Syntactic Similarity (rough scale): {syntactic_similarity:.4f}")
    else:
        print("[CHECKPOINT] Skipping syntactic-level analysis.")

    answer_lexical = input("\nDo you want to perform lexical analysis? [y|n]: ").strip().lower()
    if answer_lexical in ["y", "yes"]:
        answer_lex_type = input("Choose lexical analysis type - (s)imple or (f)ull (using NPMI seed co-occurrence): ").strip().lower()
        if answer_lex_type.startswith("f"):
            print("\n[CHECKPOINT] Performing FULL lexical analysis (NPMI-based)")
            lex_vector1 = get_lexical_style_full(text1)
            lex_vector2 = get_lexical_style_full(text2)
        else:
            print("\n[CHECKPOINT] Performing SIMPLE lexical analysis (seed counts)")
            lex_vector1 = get_lexical_style_simple(text1)
            lex_vector2 = get_lexical_style_simple(text2)
    
        print("\nLexical Style Vector (File 1):")
        for k, v in lex_vector1.items():
            print(f"  {k}: {v:.3f}")
        print("\nLexical Style Vector (File 2):")
        for k, v in lex_vector2.items():
            print(f"  {k}: {v:.3f}")
        
        # Compare the two lexical style vectors using MSE
        mse_lex, lexical_similarity = compare_lexical_vectors(lex_vector1, lex_vector2)
        print(f"\nLexical Style MSE: {mse_lex:.6f}")
        print(f"Lexical Style Similarity (1/(1+MSE)): {lexical_similarity:.4f}")
    else:
        print("[CHECKPOINT] Skipping lexical analysis.")

    print("\n[CHECKPOINT] DONE.")

if __name__ == "__main__":
    main()
