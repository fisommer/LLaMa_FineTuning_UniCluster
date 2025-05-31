# #!/usr/bin/env python3
# """
# analysis_constituency.py

# Revised script that uses Algorithm 1 from the paper to classify sentences into:
# simple, compound, complex, complex-compound, and other, based on their constituency parse trees.
# It also includes two lexical analyses: (a) a simple seed-count method and (b) a full lexical style
# analysis using normalized PMI with seed word lists for eight categories, which are then combined into
# four spectrums: subjective-objective, concrete-abstract, literary-colloquial, and formal-informal.

# After launching, you will be prompted to choose which analyses to run.
# """
# import math
# import nltk
# import re
# import spacy
# import benepar
# from nltk import Tree
# from scipy.spatial.distance import jensenshannon

# # Import the seed loader module.
# from seed_loader import load_seed_words

# # Ensure that spaCy uses Benepar for constituency parsing.
# nlp = spacy.load("en_core_web_sm")
# if "benepar" not in nlp.pipe_names:
#     nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# # Default file paths
# FILE_1 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Mark_Twain/Splits/eval_every_15th_para_sample.txt"
# FILE_2 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Charles_Dickens/Splits/eval_every_15th_para_sample.txt"

# '''
# # Define maximum expected differences for normalization
# max_diffs = {
#     "avg_words_per_sentence": 10,        # maximum expected difference in words per sentence
#     "avg_commas_per_sentence": 1,          # maximum expected difference for commas
#     "avg_semicolons_per_sentence": 0.3,       # maximum expected difference for semicolons
#     "avg_colons_per_sentence": 0.1,           # maximum expected difference for colons
#     "avg_sentences_per_paragraph": 1       # maximum expected difference for sentences per paragraph
# }

# '''
# # ---------------------------
# # Load Seed Words from CSV Files
# # ---------------------------
# seed_lists = load_seed_words()

# # Load seeds and update global seed sets:
# LITERARY_SEEDS = set(seed_lists.get("literary", []))
# ABSTRACT_SEEDS = set(seed_lists.get("abstract", []))
# OBJECTIVE_SEEDS = set(seed_lists.get("objective", []))
# COLLOQUIAL_SEEDS = set(seed_lists.get("colloquial", []))
# CONCRETE_SEEDS = set(seed_lists.get("concrete", []))
# SUBJECTIVE_SEEDS = set(seed_lists.get("subjective", []))
# FORMAL_SEEDS = set(seed_lists.get("formal", []))
# INFORMAL_SEEDS = set(seed_lists.get("informal", []))


# # ---------------------------
# # Part 1: Surface-Level Features
# # ---------------------------
# def get_surface_features(text):
#     print("[CHECKPOINT] Entering get_surface_features")
#     sentences = nltk.sent_tokenize(text)
#     num_sentences = len(sentences)
#     print(f"[CHECKPOINT] Number of sentences: {num_sentences}")
    
#     if num_sentences == 0:
#         print("[CHECKPOINT] No sentences found, returning zeros.")
#         return {
#             "avg_words_per_sentence": 0.0,
#             "avg_commas_per_sentence": 0.0,
#             "avg_semicolons_per_sentence": 0.0,
#             "avg_colons_per_sentence": 0.0,
#             "avg_sentences_per_paragraph": 0.0
#         }
    
#     total_words = sum(len(nltk.word_tokenize(sent)) for sent in sentences)
#     avg_words_per_sentence = total_words / num_sentences

#     total_commas = sum(sent.count(",") for sent in sentences)
#     total_semicolons = sum(sent.count(";") for sent in sentences)
#     total_colons = sum(sent.count(":") for sent in sentences)

#     avg_commas_per_sentence = total_commas / num_sentences
#     avg_semicolons_per_sentence = total_semicolons / num_sentences
#     avg_colons_per_sentence = total_colons / num_sentences

#     paragraphs = text.split("\n\n")
#     num_paragraphs = len(paragraphs)
#     total_sents_in_paras = sum(len(nltk.sent_tokenize(p.strip())) for p in paragraphs if p.strip())
#     avg_sentences_per_paragraph = total_sents_in_paras / num_paragraphs if num_paragraphs > 0 else 0.0

#     return {
#         "avg_words_per_sentence": avg_words_per_sentence,
#         "avg_commas_per_sentence": avg_commas_per_sentence,
#         "avg_semicolons_per_sentence": avg_semicolons_per_sentence,
#         "avg_colons_per_sentence": avg_colons_per_sentence,
#         "avg_sentences_per_paragraph": avg_sentences_per_paragraph
#     }

# def compare_surface_features(features1, features2):
#     print("[CHECKPOINT] Comparing surface features with MSE")
#     keys = [
#         "avg_words_per_sentence",
#         "avg_commas_per_sentence",
#         "avg_semicolons_per_sentence",
#         "avg_colons_per_sentence",
#         "avg_sentences_per_paragraph"
#     ]
#     squared_sum = 0.0
#     for k in keys:
#         diff = (features1[k] - features2[k]) ** 2
#         squared_sum += diff
#         print(f"[CHECKPOINT] {k}: F1={features1[k]:.3f}, F2={features2[k]:.3f}, (diff^2)={diff:.6f}")
#     mse = squared_sum / len(keys)
#     similarity = 1 / (1 + mse)
#     print(f"[CHECKPOINT] MSE for surface features = {mse:.6f}")
#     print(f"[CHECKPOINT] Surface-Level Similarity (1/(1+MSE)) = {similarity:.4f}")
#     return mse, similarity

# '''
# def compare_surface_features_normalized(features1, features2, max_diffs):
#     """
#     Compute a normalized Mean Squared Error (MSE) between two sets of surface features.
#     Each feature difference is normalized by a predetermined maximum expected difference.
#     """
#     keys = [
#         "avg_words_per_sentence",
#         "avg_commas_per_sentence",
#         "avg_semicolons_per_sentence",
#         "avg_colons_per_sentence",
#         "avg_sentences_per_paragraph"
#     ]
#     normalized_sum = 0.0
#     for k in keys:
#         # Normalize the absolute difference using the expected maximum difference for the feature.
#         norm_diff = abs(features1[k] - features2[k]) / max_diffs.get(k, 1)
#         normalized_sum += norm_diff ** 2
#         print(f"[CHECKPOINT] {k}: F1={features1[k]:.3f}, F2={features2[k]:.3f}, normalized diff^2={norm_diff**2:.6f}")
#     mse_norm = normalized_sum / len(keys)
#     # The normalized MSE is now in the range [0, 1].
#     similarity = 1 / (1 + mse_norm)
#     print(f"[CHECKPOINT] Normalized MSE = {mse_norm:.6f}")
#     print(f"[CHECKPOINT] Surface-Level Similarity (1/(1+normalized MSE)) = {similarity:.4f}")
#     return mse_norm, similarity
# '''
# # ---------------------------
# # Part 2: Syntactic-Level Features (using Algorithm 1)
# # ---------------------------
# def clean_text_for_parsing(text):
#     print("[CHECKPOINT] Starting text cleaning.")
#     text = re.sub(r'\n+', ' ', text)
#     text = re.sub(r'["“”]', '', text)
#     text = re.sub(r'[-–—]', '', text)
#     text = re.sub(r'_+', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'\s+([,;:.!?])', r'\1', text)
#     cleaned = text.strip()
#     print(f"[CHECKPOINT] Finished text cleaning. Cleaned text length: {len(cleaned)} characters.")
#     return cleaned

# def classify_sentence_benepar_safe(sentence):
#     sentence_clean = sentence.strip().replace("“", "\"").replace("”", "\"")
#     if not sentence_clean:
#         return "other"
#     try:
#         doc = nlp(sentence_clean)
#         sent_span = list(doc.sents)[0]
#         tree_str = sent_span._.parse_string
#         if not tree_str:
#             return "other"
#         tree = Tree.fromstring(tree_str)
#     except Exception:
#         return "other"

#     Ltop = [child.label() for child in tree if isinstance(child, Tree)]
#     all_labels = [subtree.label() for subtree in tree.subtrees() if isinstance(subtree, Tree)]
    
#     if "S" in Ltop:
#         if "SBAR" not in all_labels:
#             return "compound"
#         else:
#             return "complex-compound"
#     elif "VP" in Ltop:
#         if "SBAR" not in all_labels:
#             return "simple"
#         else:
#             return "complex"
#     else:
#         return "other"

# def get_syntactic_distribution(text):
#     print("[CHECKPOINT] Entering get_syntactic_distribution")
#     sentences = nltk.sent_tokenize(text)
#     print(f"[CHECKPOINT] Found {len(sentences)} sentences for syntactic analysis")
    
#     counts = {"simple": 0, "compound": 0, "complex": 0, "complex-compound": 0, "other": 0}
#     failure_count = 0
#     failed_sentences = []
    
#     for sent in sentences:
#         cat = classify_sentence_benepar_safe(sent)
#         if cat == "other":
#             failure_count += 1
#             if len(failed_sentences) < 10:
#                 failed_sentences.append(sent)
#         counts[cat] += 1

#     total_sents = len(sentences)
#     if total_sents == 0:
#         print("[CHECKPOINT] No sentences - returning zero distribution.")
#         return [0, 0, 0, 0, 0]
    
#     print(f"[CHECKPOINT] {failure_count} sentences could not be parsed reliably.")
#     if failed_sentences:
#         print("[CHECKPOINT] Sample of up to 10 sentences that could not be parsed reliably:")
#         for i, fs in enumerate(failed_sentences, start=1):
#             print(f"  {i}. {fs}")
    
#     dist = [
#         counts["simple"] / total_sents,
#         counts["compound"] / total_sents,
#         counts["complex"] / total_sents,
#         counts["complex-compound"] / total_sents,
#         counts["other"] / total_sents
#     ]
#     print("[CHECKPOINT] Syntactic distribution:", dist)
#     return dist

# def compute_syntactic_jsd(dist1, dist2):
#     print("[CHECKPOINT] Computing JSD for syntactic distribution")
#     return jensenshannon(dist1, dist2)

# # ---------------------------
# # Part 3: Lexical Analysis - Simple Seed Count
# # ---------------------------
# def get_lexical_style_simple(text):
#     print("[CHECKPOINT] Starting simple lexical analysis.")
#     tokens = nltk.word_tokenize(text.lower())
    
#     count_subj = sum(1 for token in tokens if token in SUBJECTIVE_SEEDS)
#     count_obj  = sum(1 for token in tokens if token in OBJECTIVE_SEEDS)
#     count_conc = sum(1 for token in tokens if token in CONCRETE_SEEDS)
#     count_abst = sum(1 for token in tokens if token in ABSTRACT_SEEDS)
#     count_lit  = sum(1 for token in tokens if token in LITERARY_SEEDS)
#     count_coll = sum(1 for token in tokens if token in COLLOQUIAL_SEEDS)
#     count_form = sum(1 for token in tokens if token in FORMAL_SEEDS)
#     count_infm = sum(1 for token in tokens if token in INFORMAL_SEEDS)
    
#     eps = 1e-6
#     subj_obj_score = count_subj / (count_subj + count_obj + eps)
#     conc_abst_score = count_conc / (count_conc + count_abst + eps)
#     lit_coll_score  = count_lit / (count_lit + count_coll + eps)
#     form_infm_score = count_form / (count_form + count_infm + eps)
    
#     lexical_vector = {
#         "subjective_obj": subj_obj_score,
#         "concrete_abstract": conc_abst_score,
#         "literary_colloquial": lit_coll_score,
#         "formal_informal": form_infm_score
#     }
    
#     print("[CHECKPOINT] Simple lexical style vector (values in [0,1]):")
#     for k, v in lexical_vector.items():
#         print(f"  {k}: {v:.3f}")
    
#     return lexical_vector

# # ---------------------------
# # Part 4: Lexical Analysis - Full Analysis using Normalized PMI
# # ---------------------------
# def get_lexical_style_full(text):
#     print("[CHECKPOINT] Starting full lexical analysis using NPMI.")
#     eps = 1e-6
#     sentences = nltk.sent_tokenize(text)
#     total_docs = len(sentences)
#     if total_docs == 0:
#         return {"subjective_obj": 0, "concrete_abstract": 0, "literary_colloquial": 0, "formal_informal": 0}
    
#     docs = [set(nltk.word_tokenize(sent.lower())) for sent in sentences]
#     vocabulary = set()
#     for doc in docs:
#         vocabulary.update(doc)
    
#     df = {w: sum(1 for doc in docs if w in doc) for w in vocabulary}
    
#     subj_seeds = set(w.lower() for w in SUBJECTIVE_SEEDS)
#     obj_seeds  = set(w.lower() for w in OBJECTIVE_SEEDS)
#     conc_seeds = set(w.lower() for w in CONCRETE_SEEDS)
#     abst_seeds = set(w.lower() for w in ABSTRACT_SEEDS)
#     lit_seeds  = set(w.lower() for w in LITERARY_SEEDS)
#     coll_seeds = set(w.lower() for w in COLLOQUIAL_SEEDS)
#     form_seeds = set(w.lower() for w in FORMAL_SEEDS)
#     infm_seeds = set(w.lower() for w in INFORMAL_SEEDS)
    
#     seed_p = {}
#     all_seed_words = subj_seeds.union(obj_seeds, conc_seeds, abst_seeds, lit_seeds, coll_seeds, form_seeds, infm_seeds)
#     for s in all_seed_words:
#         seed_p[s] = df.get(s, 0) / total_docs

#     def npmi(p_xy, p_x, p_y):
#         if p_xy <= 0:
#             return -1.0
#         return (math.log(p_xy / (p_x * p_y) + eps)) / (-math.log(p_xy + eps))
    
#     raw_scores = {w: {"subj": 0.0, "obj": 0.0, "conc": 0.0, "abst": 0.0, "lit": 0.0, "coll": 0.0, "form": 0.0, "infm": 0.0}
#                   for w in vocabulary}
    
#     for w in vocabulary:
#         p_w = df[w] / total_docs
#         for s in subj_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["subj"] += npmi(p_ws, p_w, seed_p.get(s, 0))
#         for s in obj_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["obj"] += npmi(p_ws, p_w, seed_p.get(s, 0))
#         for s in conc_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["conc"] += npmi(p_ws, p_w, seed_p.get(s, 0))
#         for s in abst_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["abst"] += npmi(p_ws, p_w, seed_p.get(s, 0))
#         for s in lit_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["lit"] += npmi(p_ws, p_w, seed_p.get(s, 0))
#         for s in coll_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["coll"] += npmi(p_ws, p_w, seed_p.get(s, 0))
#         for s in form_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["form"] += npmi(p_ws, p_w, seed_p.get(s, 0))
#         for s in infm_seeds:
#             co = sum(1 for doc in docs if (w in doc and s in doc))
#             p_ws = co / total_docs
#             raw_scores[w]["infm"] += npmi(p_ws, p_w, seed_p.get(s, 0))
    
#     def add_constant(score, seed_set):
#         return score + len(seed_set)
    
#     for w in vocabulary:
#         raw_scores[w]["subj"] = add_constant(raw_scores[w]["subj"], subj_seeds)
#         raw_scores[w]["obj"]  = add_constant(raw_scores[w]["obj"], obj_seeds)
#         raw_scores[w]["conc"] = add_constant(raw_scores[w]["conc"], conc_seeds)
#         raw_scores[w]["abst"] = add_constant(raw_scores[w]["abst"], abst_seeds)
#         raw_scores[w]["lit"]  = add_constant(raw_scores[w]["lit"], lit_seeds)
#         raw_scores[w]["coll"] = add_constant(raw_scores[w]["coll"], coll_seeds)
#         raw_scores[w]["form"] = add_constant(raw_scores[w]["form"], form_seeds)
#         raw_scores[w]["infm"] = add_constant(raw_scores[w]["infm"], infm_seeds)
    
#     style_vectors = {}
#     for w in vocabulary:
#         subj_obj = raw_scores[w]["subj"] / (raw_scores[w]["subj"] + raw_scores[w]["obj"] + eps)
#         conc_abst = raw_scores[w]["conc"] / (raw_scores[w]["conc"] + raw_scores[w]["abst"] + eps)
#         lit_coll  = raw_scores[w]["lit"] / (raw_scores[w]["lit"] + raw_scores[w]["coll"] + eps)
#         form_infm = raw_scores[w]["form"] / (raw_scores[w]["form"] + raw_scores[w]["infm"] + eps)
#         style_vectors[w] = (subj_obj, conc_abst, lit_coll, form_infm)
    
#     avg_style = [0.0, 0.0, 0.0, 0.0]
#     for vec in style_vectors.values():
#         avg_style[0] += vec[0]
#         avg_style[1] += vec[1]
#         avg_style[2] += vec[2]
#         avg_style[3] += vec[3]
#     n_words = len(style_vectors)
#     avg_style = [x / n_words for x in avg_style]
#     final_vector = {
#         "subjective_obj": avg_style[0],
#         "concrete_abstract": avg_style[1],
#         "literary_colloquial": avg_style[2],
#         "formal_informal": avg_style[3]
#     }
#     print("[CHECKPOINT] Full lexical style analysis vector (values in [0,1]):")
#     for k, v in final_vector.items():
#         print(f"  {k}: {v:.3f}")
#     return final_vector

# # ---------------------------
# # Part 4.5: Lexical Analysis - Compare Lexical Vectors using MSE
# # ---------------------------
# def compare_lexical_vectors(vec1, vec2):
#     print("[CHECKPOINT] Comparing lexical style vectors with MSE")
#     keys = vec1.keys()  # assuming both dictionaries have the same keys
#     squared_sum = 0.0
#     for k in keys:
#         diff = (vec1[k] - vec2[k]) ** 2
#         squared_sum += diff
#         print(f"[CHECKPOINT] {k}: V1={vec1[k]:.3f}, V2={vec2[k]:.3f}, (diff^2)={diff:.6f}")
#     mse = squared_sum / len(keys)
#     similarity = 1 / (1 + mse)
#     print(f"[CHECKPOINT] MSE for lexical style vectors = {mse:.6f}")
#     print(f"[CHECKPOINT] Lexical Style Similarity (1/(1+MSE)) = {similarity:.4f}")
#     return mse, similarity

# # ---------------------------
# # Part 5: Main Program
# # ---------------------------
# def main():
#     print(f"[CHECKPOINT] Reading File 1 from: {FILE_1}")
#     with open(FILE_1, "r", encoding="utf-8") as f1:
#         text1 = f1.read()
#     print(f"[CHECKPOINT] Reading File 2 from: {FILE_2}")
#     with open(FILE_2, "r", encoding="utf-8") as f2:
#         text2 = f2.read()

#     answer_surface = input("Do you want to perform surface-level analysis? [y|n]: ").strip().lower()
#     if answer_surface in ["y", "yes"]:
#         print("\n[CHECKPOINT] Performing surface-level analysis")
#         surface1 = get_surface_features(text1)
#         surface2 = get_surface_features(text2)
    
#         print("\nSurface-Level Features for File 1:")
#         for k, v in surface1.items():
#             print(f"  {k}: {v:.3f}")
#         print("\nSurface-Level Features for File 2:")
#         for k, v in surface2.items():
#             print(f"  {k}: {v:.3f}")
    
#         #mse_norm, surface_similarity = compare_surface_features_normalized(surface1, surface2, max_diffs)
#         #print(f"\nSurface-Level MSE: {mse_norm:.6f}")
#         mse, surface_similarity = compare_surface_features(surface1, surface2)
#         print(f"\nSurface-Level MSE: {mse:.6f}")
#         print(f"Surface-Level Similarity (naive): {surface_similarity:.4f}")
#     else:
#         print("[CHECKPOINT] Skipping surface-level analysis.")

#     answer_syntactic = input("\nDo you want to perform syntactic-level analysis? [y|n]: ").strip().lower()
#     if answer_syntactic in ["y", "yes"]:
#         print("\n[CHECKPOINT] Performing syntactic-level analysis")
#         cleaned_text1 = clean_text_for_parsing(text1)
#         cleaned_text2 = clean_text_for_parsing(text2)
    
#         dist1 = get_syntactic_distribution(cleaned_text1)
#         dist2 = get_syntactic_distribution(cleaned_text2)
    
#         print("\nSyntactic Distribution (File 1):")
#         print("  [Simple, Compound, Complex, Complex-Compound, Other]:")
#         print("  ", [f"{p:.3f}" for p in dist1])
#         print("Syntactic Distribution (File 2):")
#         print("  [Simple, Compound, Complex, Complex-Compound, Other]:")
#         print("  ", [f"{p:.3f}" for p in dist2])
    
#         syntactic_jsd = compute_syntactic_jsd(dist1, dist2)
#         print(f"\nSyntactic JSD (lower is more similar): {syntactic_jsd:.4f}")
#         syntactic_similarity = 1.0 - syntactic_jsd
#         print(f"Syntactic Similarity (rough scale): {syntactic_similarity:.4f}")
#     else:
#         print("[CHECKPOINT] Skipping syntactic-level analysis.")

#     answer_lexical = input("\nDo you want to perform lexical analysis? [y|n]: ").strip().lower()
#     if answer_lexical in ["y", "yes"]:
#         answer_lex_type = input("Choose lexical analysis type - (s)imple or (f)ull (using NPMI seed co-occurrence): ").strip().lower()
#         if answer_lex_type.startswith("f"):
#             print("\n[CHECKPOINT] Performing FULL lexical analysis (NPMI-based)")
#             lex_vector1 = get_lexical_style_full(text1)
#             lex_vector2 = get_lexical_style_full(text2)
#         else:
#             print("\n[CHECKPOINT] Performing SIMPLE lexical analysis (seed counts)")
#             lex_vector1 = get_lexical_style_simple(text1)
#             lex_vector2 = get_lexical_style_simple(text2)
    
#         print("\nLexical Style Vector (File 1):")
#         for k, v in lex_vector1.items():
#             print(f"  {k}: {v:.3f}")
#         print("\nLexical Style Vector (File 2):")
#         for k, v in lex_vector2.items():
#             print(f"  {k}: {v:.3f}")
        
#         # Compare the two lexical style vectors using MSE
#         mse_lex, lexical_similarity = compare_lexical_vectors(lex_vector1, lex_vector2)
#         print(f"\nLexical Style MSE: {mse_lex:.6f}")
#         print(f"Lexical Style Similarity (1/(1+MSE)): {lexical_similarity:.4f}")
#     else:
#         print("[CHECKPOINT] Skipping lexical analysis.")

#     print("\n[CHECKPOINT] DONE.")

# if __name__ == "__main__":
#     main()


























# #!/usr/bin/env python3
# """
# analysis_constituency.py

# Revised script that uses Algorithm 1 from the paper to classify sentences into:
# simple, compound, complex, complex-compound, and other, based on their constituency parse trees.
# It also includes two lexical analyses: (a) a simple seed-count method and (b) a full lexical style
# analysis using normalized PMI with seed word lists for eight categories, which are then combined into
# four spectrums: subjective-objective, concrete-abstract, literary-colloquial, and formal-informal.

# After launching, you will be prompted to choose which analyses to run.
# """
# import math
# import nltk
# import re
# import spacy
# import nltk
# import benepar # Ensure benepar is installed: pip install benepar
# from nltk import Tree
# from scipy.spatial.distance import jensenshannon
# import logging # Added for logging
# import os # Added for os.path and os.makedirs

# # Default file paths
# FILE_1 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Mark_Twain/Splits/eval_every_15th_para_sample.txt"
# FILE_2 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Charles_Dickens/Splits/eval_every_15th_para_sample.txt"



# # Import the seed loader module.
# # This assumes seed_loader.py is in the same directory or accessible via PYTHONPATH
# try:
#     from seed_loader import load_seed_words
# except ImportError:
#     # Mock load_seed_words if not found, to allow script to run for other parts if needed
#     def load_seed_words():
#         print("WARNING: seed_loader.py not found. Lexical analysis will use empty seed lists.")
#         return {}
#     print("WARNING: Could not import load_seed_words from seed_loader.py. Lexical functions might not work as expected.")


# # --- Setup Logging ---
# # Define the target log directory
# LOG_DIRECTORY = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Style_Analysis/Output/Log_Files"
# LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "analysis_constituency.log")

# # Create the log directory if it doesn't exist
# os.makedirs(LOG_DIRECTORY, exist_ok=True)

# # Configure logging once at the beginning
# # Get the root logger
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO) # Set overall level

# # Remove any existing handlers to avoid duplicate logs if script is re-run in same session/module reloaded
# for handler in root_logger.handlers[:]:
#     root_logger.removeHandler(handler)

# # Console handler
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# root_logger.addHandler(ch)

# # File handler
# try:
#     fh = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8') # 'a' for append
#     fh.setLevel(logging.INFO) # Or logging.DEBUG for more verbose file logs
#     fh.setFormatter(formatter)
#     root_logger.addHandler(fh)
# except Exception as e:
#     print(f"Error setting up file logger at {LOG_FILE_PATH}: {e}")


# logger = logging.getLogger(__name__) # Logger for this specific module

# # --- Global Variables and Model Loading ---
# nlp_model = None # spaCy model will be loaded on demand
# PUNKT_AVAILABLE = False # Global flag for NLTK punkt resource


# max_diffs = {
#     "avg_words_per_sentence": 10.0,
#     "avg_commas_per_sentence": 2.0,
#     "avg_semicolons_per_sentence": 0.5,
#     "avg_colons_per_sentence": 0.3,
#     "avg_sentences_per_paragraph": 3.0
# }

# # ---------------------------
# # Load Seed Words
# # ---------------------------
# try:
#     seed_lists = load_seed_words()
# except Exception as e:
#     logger.error(f"Error during load_seed_words call: {e}", exc_info=True)
#     seed_lists = {} # Fallback

# LITERARY_SEEDS = set(seed_lists.get("literary", []))
# ABSTRACT_SEEDS = set(seed_lists.get("abstract", []))
# OBJECTIVE_SEEDS = set(seed_lists.get("objective", []))
# COLLOQUIAL_SEEDS = set(seed_lists.get("colloquial", []))
# CONCRETE_SEEDS = set(seed_lists.get("concrete", []))
# SUBJECTIVE_SEEDS = set(seed_lists.get("subjective", []))
# FORMAL_SEEDS = set(seed_lists.get("formal", []))
# INFORMAL_SEEDS = set(seed_lists.get("informal", []))

# # --- Helper Functions for Resource Checking ---
# def ensure_nltk_punkt():
#     global PUNKT_AVAILABLE
#     if PUNKT_AVAILABLE:
#         return True
#     try:
#         nltk.data.find('tokenizers/punkt')
#         logger.info("NLTK 'punkt' resource found.")
#         PUNKT_AVAILABLE = True
#         return True
#     except LookupError:
#         logger.warning("NLTK 'punkt' resource not found. Attempting to download...")
#         try:
#             nltk.download('punkt', quiet=True)
#             nltk.data.find('tokenizers/punkt') # Verify download
#             logger.info("NLTK 'punkt' resource downloaded successfully.")
#             PUNKT_AVAILABLE = True
#             return True
#         except Exception as e:
#             logger.error(f"Failed to download NLTK 'punkt' resource: {e}", exc_info=True)
#             logger.error("Please try downloading it manually in a Python interpreter:\n>>> import nltk\n>>> nltk.download('punkt')")
#             PUNKT_AVAILABLE = False
#             return False

# def load_nlp_model():
#     global nlp_model
#     if nlp_model is None:
#         logger.info("Loading spaCy model (en_core_web_sm) and Benepar...")
#         try:
#             nlp_model = spacy.load("en_core_web_sm")
#             if "benepar" not in nlp_model.pipe_names:
#                 logger.info("Adding Benepar pipe to spaCy model...")
#                 # Ensure benepar model is downloaded. In a Python interpreter:
#                 # import benepar
#                 # benepar.download('benepar_en3')
#                 nlp_model.add_pipe("benepar", config={"model": "benepar_en3"})
#             logger.info("spaCy model and Benepar loaded successfully.")
#         except ImportError:
#             logger.error("Benepar library not found. Please install it: pip install benepar")
#             raise
#         except OSError as e:
#             logger.error(f"Failed to load spaCy model 'en_core_web_sm' or Benepar model 'benepar_en3'. They might not be downloaded. Error: {e}")
#             logger.error("Try:\n1. python -m spacy download en_core_web_sm\n2. In Python: import benepar; benepar.download('benepar_en3')")
#             raise
#     return nlp_model

# # ---------------------------
# # Part 1: Surface-Level Features
# # ---------------------------
# def get_surface_features(text):
#     logger.info("Calculating surface features...")
#     if not ensure_nltk_punkt():
#         logger.error("NLTK 'punkt' is unavailable. Cannot calculate surface features.")
#         return {key: 0.0 for key in ["avg_words_per_sentence", "avg_commas_per_sentence", "avg_semicolons_per_sentence", "avg_colons_per_sentence", "avg_sentences_per_paragraph"]}

#     sentences = nltk.sent_tokenize(text)
#     num_sentences = len(sentences)
#     logger.info(f"Number of sentences: {num_sentences}")

#     if num_sentences == 0:
#         logger.warning("No sentences found, returning zeros for surface features.")
#         return {key: 0.0 for key in ["avg_words_per_sentence", "avg_commas_per_sentence", "avg_semicolons_per_sentence", "avg_colons_per_sentence", "avg_sentences_per_paragraph"]}

#     total_words = sum(len(nltk.word_tokenize(sent)) for sent in sentences)
#     avg_words_per_sentence = total_words / num_sentences if num_sentences > 0 else 0.0

#     total_commas = sum(sent.count(",") for sent in sentences)
#     total_semicolons = sum(sent.count(";") for sent in sentences)
#     total_colons = sum(sent.count(":") for sent in sentences)

#     avg_commas_per_sentence = total_commas / num_sentences if num_sentences > 0 else 0.0
#     avg_semicolons_per_sentence = total_semicolons / num_sentences if num_sentences > 0 else 0.0
#     avg_colons_per_sentence = total_colons / num_sentences if num_sentences > 0 else 0.0

#     paragraphs = re.split(r'\n\s*\n', text.strip())
#     num_paragraphs = len([p for p in paragraphs if p.strip()]) # Count non-empty paragraphs
    
#     if num_paragraphs > 0:
#         total_sents_in_paras = sum(len(nltk.sent_tokenize(p.strip())) for p in paragraphs if p.strip())
#         avg_sentences_per_paragraph = total_sents_in_paras / num_paragraphs
#     else:
#         avg_sentences_per_paragraph = 0.0
#         logger.warning("No paragraphs found for calculating avg_sentences_per_paragraph.")

#     logger.info("Surface features calculation complete.")
#     return {
#         "avg_words_per_sentence": avg_words_per_sentence,
#         "avg_commas_per_sentence": avg_commas_per_sentence,
#         "avg_semicolons_per_sentence": avg_semicolons_per_sentence,
#         "avg_colons_per_sentence": avg_colons_per_sentence,
#         "avg_sentences_per_paragraph": avg_sentences_per_paragraph
#     }

# def compare_surface_features_normalized(features1, features2, p_max_diffs):
#     logger.info("Comparing surface features using normalized absolute differences.")
#     keys = [
#         "avg_words_per_sentence", "avg_commas_per_sentence",
#         "avg_semicolons_per_sentence", "avg_colons_per_sentence",
#         "avg_sentences_per_paragraph"
#     ]
#     sum_normalized_abs_diff = 0.0
#     num_valid_features = 0

#     for k in keys:
#         if k in features1 and k in features2:
#             max_d = p_max_diffs.get(k)
#             if max_d is not None and max_d > 0:
#                 norm_abs_diff = abs(features1[k] - features2[k]) / max_d
#                 sum_normalized_abs_diff += norm_abs_diff
#                 num_valid_features += 1
#                 logger.debug(f"{k}: F1={features1[k]:.3f}, F2={features2[k]:.3f}, max_diff={max_d}, norm_abs_diff={norm_abs_diff:.6f}")
#             else:
#                 logger.warning(f"Feature '{k}' skipped: max_diff not defined or not positive in p_max_diffs.")
#         else:
#             logger.warning(f"Feature '{k}' skipped: not present in one or both feature sets.")

#     if num_valid_features == 0:
#         logger.error("No valid features for normalized surface comparison. Returning max divergence.")
#         return float('inf'), 0.0

#     avg_norm_abs_diff = sum_normalized_abs_diff / num_valid_features
#     similarity = 1.0 / (1.0 + avg_norm_abs_diff)
#     logger.info(f"Average Normalized Absolute Difference for surface features = {avg_norm_abs_diff:.6f}")
#     logger.info(f"Surface-Level Similarity (1/(1+AvgNormAbsDiff)) = {similarity:.4f}")
#     return avg_norm_abs_diff, similarity

# def compare_surface_features_mse(features1, features2):
#     logger.info("Comparing surface features with MSE (Mean Squared Error).")
#     keys = [
#         "avg_words_per_sentence", "avg_commas_per_sentence",
#         "avg_semicolons_per_sentence", "avg_colons_per_sentence",
#         "avg_sentences_per_paragraph"
#     ]
#     squared_sum = 0.0
#     num_valid_features = 0
#     for k in keys:
#         if k in features1 and k in features2:
#             diff = (features1[k] - features2[k]) ** 2
#             squared_sum += diff
#             num_valid_features +=1
#             logger.debug(f"{k}: F1={features1[k]:.3f}, F2={features2[k]:.3f}, (diff^2)={diff:.6f}")
#         else:
#             logger.warning(f"Feature '{k}' skipped: not present in one or both feature sets.")

#     if num_valid_features == 0:
#         logger.error("No valid features for MSE surface comparison. Returning max divergence.")
#         return float('inf'), 0.0

#     mse = squared_sum / num_valid_features
#     similarity = 1 / (1 + mse)
#     logger.info(f"MSE for surface features = {mse:.6f}")
#     logger.info(f"Surface-Level Similarity (1/(1+MSE)) = {similarity:.4f}")
#     return mse, similarity

# # ---------------------------
# # Part 2: Syntactic-Level Features
# # ---------------------------
# def clean_text_for_parsing(text):
#     # This function is not strictly needed if individual sentences are cleaned before parsing.
#     # However, if a whole block of text is passed to nlp(), this pre-cleaning might be useful.
#     logger.info("Starting text cleaning for parsing (block level).")
#     text = re.sub(r'\s*\n\s*', ' ', text)
#     text = re.sub(r'["“”]', '', text)
#     text = re.sub(r'[-–—]', ' ', text)
#     text = re.sub(r'_+', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'\s+([,;:.!?])', r'\1', text)
#     cleaned = text.strip()
#     logger.info(f"Finished block-level text cleaning. Length: {len(cleaned)} chars.")
#     return cleaned

# def classify_sentence_benepar_safe(sentence_text):
#     nlp_instance = load_nlp_model() # Ensure model is loaded
#     sentence_clean = sentence_text.strip() # Already cleaned individually in get_syntactic_distribution

#     if not sentence_clean:
#         logger.debug("Empty sentence string provided to classify_sentence_benepar_safe.")
#         return "other"

#     try:
#         doc = nlp_instance(sentence_clean)
#         if not list(doc.sents):
#             logger.warning(f"spaCy did not segment sentence: '{sentence_clean[:100]}...'")
#             return "other"
#         sent_span = list(doc.sents)[0] # Assume the input is one logical sentence
#         tree_str = sent_span._.parse_string
#         if not tree_str:
#             logger.warning(f"Benepar failed to produce parse string for: '{sent_span.text[:100]}...'")
#             return "other"
#         tree = Tree.fromstring(tree_str)
#     except Exception as e:
#         logger.error(f"Error parsing sentence '{sentence_clean[:100]}...': {e}", exc_info=False)
#         return "other"

#     Ltop = [child.label() for child in tree if isinstance(child, Tree)]
#     all_constituent_labels = [subtree.label() for subtree in tree.subtrees() if isinstance(subtree, Tree)]
    
#     # Heuristic for classification (can be refined based on linguistic theory or Algorithm 1 specifics)
#     has_sbar = "SBAR" in all_constituent_labels
#     # Count S nodes that are direct children of the root (or one level down if ROOT is implicit)
#     # This is a simplification; true clause counting is more complex.
#     num_main_clauses_approx = 0
#     if tree.label() == "S" or (tree.label().lower() == "root" and "S" in Ltop) : # Common for full sentence parses
#         num_main_clauses_approx = 1 # Assume at least one main clause if S is high up
#         # Check for co-ordinating conjunctions (CC) between S nodes at a similar level for compound
#         # This requires deeper tree traversal than simple Ltop.
#         # For now, we simplify: if multiple S at Ltop, or S + CC + S pattern.
#         if Ltop.count("S") > 1 or ("CC" in Ltop and Ltop.count("S") >=1 ): # Basic check
#              num_main_clauses_approx = Ltop.count("S") # Rough estimate

#     if has_sbar:
#         if num_main_clauses_approx > 1:
#             return "complex-compound"
#         elif num_main_clauses_approx == 1:
#             return "complex"
#         else: # SBAR exists, but main clause structure unclear by this simple check (e.g. fragment with SBAR)
#             return "complex" # Default to complex if SBAR present but main clause count ambiguous
#     else: # No SBAR
#         if num_main_clauses_approx > 1:
#             return "compound"
#         elif num_main_clauses_approx == 1 or (not Ltop and tree.label() == "VP"): # A single S or just a VP (imperative)
#             return "simple"
#         elif "VP" in Ltop and not Ltop.count("S"): # Possibly simple imperative
#             return "simple"

#     logger.debug(f"Sentence '{sentence_text[:50]}...' classified as 'other'. Root: {tree.label()}, Ltop: {Ltop}, Has SBAR: {has_sbar}")
#     return "other"


# def get_syntactic_distribution(text):
#     logger.info("Calculating syntactic distribution of sentence types...")
#     if not ensure_nltk_punkt():
#         logger.error("NLTK 'punkt' is unavailable. Cannot calculate syntactic distribution.")
#         return [0.0, 0.0, 0.0, 0.0, 0.0]

#     sentences = nltk.sent_tokenize(text)
#     logger.info(f"Found {len(sentences)} sentences for syntactic analysis.")

#     counts = {"simple": 0, "compound": 0, "complex": 0, "complex-compound": 0, "other": 0}
#     failure_count = 0
#     parsed_sentence_count = 0
#     failed_sentence_samples = []

#     for i, sent_text in enumerate(sentences):
#         original_sent_text = sent_text # Keep original for logging if needed
#         cleaned_sent_for_parser = re.sub(r'\s*\n\s*', ' ', sent_text).strip()
#         cleaned_sent_for_parser = re.sub(r'["“”]', '', cleaned_sent_for_parser)
#         cleaned_sent_for_parser = re.sub(r'[-–—]', ' ', cleaned_sent_for_parser)
#         cleaned_sent_for_parser = re.sub(r'_+', '', cleaned_sent_for_parser)
#         cleaned_sent_for_parser = re.sub(r'\s+', ' ', cleaned_sent_for_parser)
#         cleaned_sent_for_parser = re.sub(r'\s+([,;:.!?])', r'\1', cleaned_sent_for_parser).strip()

#         if not cleaned_sent_for_parser:
#             logger.debug(f"Sentence {i+1} became empty after cleaning: '{original_sent_text[:50]}...'")
#             continue

#         parsed_sentence_count +=1
#         cat = classify_sentence_benepar_safe(cleaned_sent_for_parser)
#         counts[cat] += 1
#         if cat == "other":
#             failure_count += 1
#             if len(failed_sentence_samples) < 10:
#                 failed_sentence_samples.append(cleaned_sent_for_parser)
        
#         if (i + 1) % 100 == 0 and i > 0 :
#             logger.info(f"Processed {i+1}/{len(sentences)} sentences for syntax...")


#     if parsed_sentence_count == 0:
#         logger.warning("No sentences processed for syntactic distribution.")
#         return [0.0, 0.0, 0.0, 0.0, 0.0]

#     logger.info(f"{failure_count}/{parsed_sentence_count} sentences classified as 'other' (includes parse failures).")
#     if failed_sentence_samples:
#         logger.info("Sample of up to 10 sentences classified as 'other' or failed to parse reliably:")
#         for idx, fs_sample in enumerate(failed_sentence_samples, start=1):
#             logger.info(f"  Sample {idx}: {fs_sample[:150]}...")

#     dist_order = ["simple", "compound", "complex", "complex-compound", "other"]
#     dist = [counts[cat_name] / parsed_sentence_count for cat_name in dist_order]
    
#     log_dist_str = ", ".join([f"{cat_name.capitalize()}={dist[j]:.3f}" for j, cat_name in enumerate(dist_order)])
#     logger.info(f"Syntactic distribution: {log_dist_str}")
#     return dist

# def compute_syntactic_jsd(dist1, dist2):
#     logger.info("Computing Jensen-Shannon Divergence (JSD) for syntactic distributions.")
#     import numpy as np # JSD function expects numpy arrays
#     d1_np = np.array(dist1, dtype=float)
#     d2_np = np.array(dist2, dtype=float)

#     # Ensure distributions sum to 1 for JSD calculation (add small epsilon and re-normalize)
#     eps_norm = 1e-9
#     if not np.isclose(np.sum(d1_np), 1.0) and np.sum(d1_np) > 0:
#         d1_np = (d1_np + eps_norm) / np.sum(d1_np + eps_norm * len(d1_np))
#     if not np.isclose(np.sum(d2_np), 1.0) and np.sum(d2_np) > 0:
#         d2_np = (d2_np + eps_norm) / np.sum(d2_np + eps_norm * len(d2_np))
    
#     # Handle cases where a distribution might be all zeros after normalization attempt if original was all zero
#     if np.sum(d1_np) == 0 or np.sum(d2_np) == 0:
#         logger.warning("One or both syntactic distributions are zero. JSD will be maximal or undefined.")
#         if np.array_equal(d1_np, d2_np): return 0.0 # Both zero, so identical
#         return 1.0 # Max divergence if one is zero and other is not (or both different zeros)

#     try:
#         jsd = jensenshannon(d1_np, d2_np, base=2.0)
#         logger.info(f"Calculated JSD: {jsd:.6f}")
#         return jsd
#     except ValueError as ve:
#         logger.error(f"ValueError during JSD calculation (often due to negative values or non-finite inputs): {ve}")
#         logger.error(f"Dist1: {d1_np}, Sum: {np.sum(d1_np)}")
#         logger.error(f"Dist2: {d2_np}, Sum: {np.sum(d2_np)}")
#         return 1.0 # Return max divergence on error


# # ---------------------------
# # Part 3: Lexical Analysis - Simple Seed Count
# # ---------------------------
# def get_lexical_style_simple(text):
#     logger.info("Starting simple lexical analysis (seed counts).")
#     if not ensure_nltk_punkt():
#         logger.error("NLTK 'punkt' is unavailable. Cannot perform simple lexical analysis.")
#         return {"subjective_obj": 0.0, "concrete_abstract": 0.0, "literary_colloquial": 0.0, "formal_informal": 0.0}

#     tokens = nltk.word_tokenize(text.lower())
#     if not tokens:
#         logger.warning("No tokens found for simple lexical analysis.")
#         return {"subjective_obj": 0.0, "concrete_abstract": 0.0, "literary_colloquial": 0.0, "formal_informal": 0.0}

#     # Counts
#     counts = {cat: sum(1 for token in tokens if token in s_list) for cat, s_list in {
#         "subj": SUBJECTIVE_SEEDS, "obj": OBJECTIVE_SEEDS, "conc": CONCRETE_SEEDS,
#         "abst": ABSTRACT_SEEDS, "lit": LITERARY_SEEDS, "coll": COLLOQUIAL_SEEDS,
#         "form": FORMAL_SEEDS, "infm": INFORMAL_SEEDS
#     }.items()}

#     eps = 1e-9
#     lexical_vector = {
#         "subjective_obj": counts["subj"] / (counts["subj"] + counts["obj"] + eps),
#         "concrete_abstract": counts["conc"] / (counts["conc"] + counts["abst"] + eps),
#         "literary_colloquial": counts["lit"] / (counts["lit"] + counts["coll"] + eps),
#         "formal_informal": counts["form"] / (counts["form"] + counts["infm"] + eps)
#     }
#     logger.info(f"Simple lexical style vector: {lexical_vector}")
#     return lexical_vector

# # ---------------------------
# # Part 4: Lexical Analysis - Full Analysis using Normalized PMI
# # ---------------------------
# def get_lexical_style_full(text):
#     logger.info("Starting full lexical analysis using NPMI. This may take some time...")
#     eps = 1e-9

#     if not ensure_nltk_punkt():
#         logger.error("NLTK 'punkt' is unavailable. Cannot perform full lexical analysis.")
#         return {"subjective_obj": 0.0, "concrete_abstract": 0.0, "literary_colloquial": 0.0, "formal_informal": 0.0}

#     sentences = nltk.sent_tokenize(text)
#     docs = [set(nltk.word_tokenize(sent.lower())) for sent in sentences if sent.strip()]
#     docs = [doc for doc in docs if doc] # Filter out empty sets from empty sentences
#     total_docs = len(docs)

#     if total_docs == 0:
#         logger.warning("No sentences (documents) found for full lexical analysis after tokenization.")
#         return {"subjective_obj": 0.0, "concrete_abstract": 0.0, "literary_colloquial": 0.0, "formal_informal": 0.0}

#     vocabulary = set.union(*docs) if docs else set()
#     if not vocabulary:
#         logger.warning("Vocabulary is empty for full lexical analysis.")
#         return {"subjective_obj": 0.0, "concrete_abstract": 0.0, "literary_colloquial": 0.0, "formal_informal": 0.0}

#     df = {w: sum(1 for doc_set in docs if w in doc_set) for w in vocabulary}

#     seed_categories_map = {
#         "subj": SUBJECTIVE_SEEDS, "obj": OBJECTIVE_SEEDS, "conc": CONCRETE_SEEDS,
#         "abst": ABSTRACT_SEEDS, "lit": LITERARY_SEEDS, "coll": COLLOQUIAL_SEEDS,
#         "form": FORMAL_SEEDS, "infm": INFORMAL_SEEDS
#     }
#     active_seed_words_by_cat = {
#         cat_name: set(w.lower() for w in s_list if w.lower() in vocabulary)
#         for cat_name, s_list in seed_categories_map.items()
#     }

#     p_seed = {}
#     for cat_name, s_set in active_seed_words_by_cat.items():
#         for s_word in s_set:
#             if s_word not in p_seed:
#                  p_seed[s_word] = (df.get(s_word, 0) + eps) / (total_docs + eps) # Smoothed P(seed)

#     def calculate_npmi(p_xy, p_x, p_y, smoothing_eps=1e-12): # using a smaller eps for npmi internals
#         # Ensure probabilities are not zero to avoid math domain errors with log
#         p_xy = max(p_xy, smoothing_eps)
#         p_x = max(p_x, smoothing_eps)
#         p_y = max(p_y, smoothing_eps)

#         log_p_xy = math.log(p_xy, 2)
        
#         # PMI = log2( P(x,y) / (P(x)*P(y)) )
#         try:
#             pmi_val = math.log(p_xy / (p_x * p_y), 2)
#         except ValueError: # Can happen if p_x * p_y is zero or negative, or ratio is zero/negative
#             pmi_val = -float('inf') # Max negative association if arguments are problematic

#         if p_xy == 1.0: # if P(x,y) is 1 (e.g. x and y are same word and appear in all docs)
#             return 1.0 if pmi_val >=0 else -1.0 # NPMI is 1 if PMI is positive, -1 if PMI is negative

#         if log_p_xy == 0: # Should be caught by p_xy == 1.0
#              return 1.0 if pmi_val >=0 else -1.0
        
#         npmi_val = pmi_val / (-log_p_xy)
#         return max(-1.0, min(1.0, npmi_val))

#     raw_word_scores = {w: {cat_name: 0.0 for cat_name in seed_categories_map} for w in vocabulary}

#     for w_idx, w in enumerate(vocabulary):
#         if (w_idx + 1) % 200 == 0: # Log progress less frequently
#             logger.info(f"NPMI processing: word {w_idx+1}/{len(vocabulary)} ('{w}')")
        
#         p_w = (df.get(w, 0) + eps) / (total_docs + eps)
#         if p_w <= eps/total_docs : continue # Effectively zero probability

#         for cat_name, current_cat_seed_set in active_seed_words_by_cat.items():
#             category_npmi_sum = 0.0
#             num_seeds_contributing = 0
#             for s_word in current_cat_seed_set:
#                 if s_word == w: continue

#                 co_occurrence_count = sum(1 for doc_set in docs if w in doc_set and s_word in doc_set)
#                 p_ws = (co_occurrence_count + eps) / (total_docs + eps)
                
#                 current_p_seed_s = p_seed.get(s_word, eps / total_docs) # Smoothed P(s)
                
#                 npmi_val = calculate_npmi(p_ws, p_w, current_p_seed_s)
#                 category_npmi_sum += npmi_val
#                 num_seeds_contributing += 1
            
#             if num_seeds_contributing > 0:
#                 raw_word_scores[w][cat_name] = category_npmi_sum / num_seeds_contributing
#             else:
#                 raw_word_scores[w][cat_name] = 0.0 # Neutral score if no active seeds for this cat/word combo

#     word_style_spectrums = {w: {} for w in vocabulary}
#     for w in vocabulary:
#         s_subj_shifted = raw_word_scores[w]["subj"] + 1.0
#         s_obj_shifted  = raw_word_scores[w]["obj"]  + 1.0
#         s_conc_shifted = raw_word_scores[w]["conc"] + 1.0
#         s_abst_shifted = raw_word_scores[w]["abst"] + 1.0
#         s_lit_shifted  = raw_word_scores[w]["lit"]  + 1.0
#         s_coll_shifted = raw_word_scores[w]["coll"] + 1.0
#         s_form_shifted = raw_word_scores[w]["form"] + 1.0
#         s_infm_shifted = raw_word_scores[w]["infm"] + 1.0

#         word_style_spectrums[w]["subjective_obj"]     = s_subj_shifted / (s_subj_shifted + s_obj_shifted + eps)
#         word_style_spectrums[w]["concrete_abstract"]  = s_conc_shifted / (s_conc_shifted + s_abst_shifted + eps)
#         word_style_spectrums[w]["literary_colloquial"]= s_lit_shifted  / (s_lit_shifted  + s_coll_shifted + eps)
#         word_style_spectrums[w]["formal_informal"]    = s_form_shifted / (s_form_shifted + s_infm_shifted + eps)

#     avg_style_spectrum = {key: 0.0 for key in word_style_spectrums.get(next(iter(vocabulary), {}), {}).keys()}
#     if not vocabulary: return avg_style_spectrum # Should be caught earlier

#     for w_spectrum_scores in word_style_spectrums.values():
#         for key, val in w_spectrum_scores.items():
#             avg_style_spectrum[key] += val

#     num_vocab_words = len(vocabulary)
#     final_text_lexical_vector = {key: val / num_vocab_words for key, val in avg_style_spectrum.items()}
#     logger.info(f"Full lexical style vector: {final_text_lexical_vector}")
#     return final_text_lexical_vector

# # ---------------------------
# # Part 4.5: Lexical Analysis - Compare Lexical Vectors using MSE
# # ---------------------------
# def compare_lexical_vectors(vec1, vec2):
#     logger.info("Comparing lexical style vectors with MSE.")
#     # Ensure keys are consistent, take from vec1, assume vec2 has same or will default to 0
#     keys = vec1.keys()
#     if not keys:
#         logger.warning("Empty lexical vectors for comparison.")
#         return float('inf'), 0.0

#     squared_sum = 0.0
#     for k in keys:
#         val1 = vec1.get(k, 0.0)
#         val2 = vec2.get(k, 0.0)
#         diff = (val1 - val2) ** 2
#         squared_sum += diff
#         logger.debug(f"{k}: V1={val1:.3f}, V2={val2:.3f}, (diff^2)={diff:.6f}")

#     if not keys: # Should not happen if previous check passed, but defensive
#         return float('inf'), 0.0
        
#     mse = squared_sum / len(keys)
#     similarity = 1 / (1 + mse)
#     logger.info(f"MSE for lexical style vectors = {mse:.6f}")
#     logger.info(f"Lexical Style Similarity (1/(1+MSE)) = {similarity:.4f}")
#     return mse, similarity

# # ---------------------------
# # Part 5: Main Program
# # ---------------------------
# def main():
#     logger.info(f"--- Starting Analysis ---")
#     logger.info(f"Reading File 1 from: {FILE_1}")
#     try:
#         with open(FILE_1, "r", encoding="utf-8") as f1: text1 = f1.read()
#     except FileNotFoundError: logger.error(f"File not found: {FILE_1}"); return
#     except Exception as e: logger.error(f"Error reading {FILE_1}: {e}"); return

#     logger.info(f"Reading File 2 from: {FILE_2}")
#     try:
#         with open(FILE_2, "r", encoding="utf-8") as f2: text2 = f2.read()
#     except FileNotFoundError: logger.error(f"File not found: {FILE_2}"); return
#     except Exception as e: logger.error(f"Error reading {FILE_2}: {e}"); return

#     # --- Surface-Level Analysis ---
#     answer_surface = input("Perform surface-level analysis? [y|n]: ").strip().lower()
#     if answer_surface.startswith("y"):
#         logger.info("User requested surface-level analysis.")
#         surface1 = get_surface_features(text1)
#         surface2 = get_surface_features(text2)
#         print("\nSurface-Level Features for File 1:")
#         for k, v in surface1.items(): print(f"  {k}: {v:.3f}")
#         print("\nSurface-Level Features for File 2:")
#         for k, v in surface2.items(): print(f"  {k}: {v:.3f}")
#         norm_abs_diff, surface_similarity = compare_surface_features_normalized(surface1, surface2, max_diffs)
#         print(f"\nSurface-Level Average Normalized Absolute Difference: {norm_abs_diff:.6f}")
#         print(f"Surface-Level Similarity (Normalized Method): {surface_similarity:.4f}")
#     else:
#         logger.info("Skipping surface-level analysis.")

#     # --- Syntactic-Level Analysis ---
#     answer_syntactic = input("\nPerform syntactic-level analysis? [y|n]: ").strip().lower()
#     if answer_syntactic.startswith("y"):
#         logger.info("User requested syntactic-level analysis.")
#         try:
#             load_nlp_model() # Ensure models are ready
#             dist1 = get_syntactic_distribution(text1)
#             dist2 = get_syntactic_distribution(text2)
#             print("\nSyntactic Distribution (File 1) [Sim, Com, Cx, Cx-Com, Oth]:")
#             print(f"  {[f'{p:.3f}' for p in dist1]}")
#             print("Syntactic Distribution (File 2) [Sim, Com, Cx, Cx-Com, Oth]:")
#             print(f"  {[f'{p:.3f}' for p in dist2]}")
#             syntactic_jsd = compute_syntactic_jsd(dist1, dist2)
#             print(f"\nSyntactic JSD (lower is more similar): {syntactic_jsd:.4f}")
#             syntactic_similarity = 1.0 - syntactic_jsd
#             print(f"Syntactic Similarity (1 - JSD): {syntactic_similarity:.4f}")
#         except Exception as e:
#             logger.error(f"Error during syntactic analysis: {e}", exc_info=True)
#             print("ERROR during syntactic analysis. Check logs.")
#     else:
#         logger.info("Skipping syntactic-level analysis.")

#     # --- Lexical Analysis ---
#     answer_lexical = input("\nPerform lexical analysis? [y|n]: ").strip().lower()
#     if answer_lexical.startswith("y"):
#         logger.info("User requested lexical analysis.")
#         if not any(LITERARY_SEEDS) and not any(ABSTRACT_SEEDS): # Basic check if seeds loaded
#              logger.warning("Seed lists appear empty. Lexical analysis might not be meaningful.")
#         answer_lex_type = input("Choose lexical analysis type - (s)imple seed count or (f)ull NPMI: ").strip().lower()
#         if answer_lex_type.startswith("f"):
#             logger.info("Performing FULL lexical analysis (NPMI-based).")
#             lex_vector1 = get_lexical_style_full(text1)
#             lex_vector2 = get_lexical_style_full(text2)
#         else:
#             logger.info("Performing SIMPLE lexical analysis (seed counts).")
#             lex_vector1 = get_lexical_style_simple(text1)
#             lex_vector2 = get_lexical_style_simple(text2)
#         print("\nLexical Style Vector (File 1):")
#         for k, v in lex_vector1.items(): print(f"  {k}: {v:.3f}")
#         print("\nLexical Style Vector (File 2):")
#         for k, v in lex_vector2.items(): print(f"  {k}: {v:.3f}")
#         mse_lex, lexical_similarity = compare_lexical_vectors(lex_vector1, lex_vector2)
#         print(f"\nLexical Style MSE: {mse_lex:.6f}")
#         print(f"Lexical Style Similarity (1/(1+MSE)): {lexical_similarity:.4f}")
#     else:
#         logger.info("Skipping lexical analysis.")

#     logger.info("--- Analysis script finished. ---")

# if __name__ == "__main__":
#     # This initial call ensures NLTK data is checked/downloaded once at script start if needed by any function
#     ensure_nltk_punkt()
#     main()















#!/usr/bin/env python3
"""
analysis_constituency.py

Implements stylistic analysis metrics inspired by Syed et al. (2020) and
utilizes syntactic sentence classification based on Feng, Banerjee, and Choi (2012).

Metrics include:
- Surface-level: Avg words/sentence, punctuation counts, avg sentences/paragraph.
  Comparison via MSE (as per Syed et al.) or optionally Normalized Absolute Difference.
- Syntactic-level: Distribution over Simple, Compound, Complex, Complex-Compound, Other.
  Classification based on Feng et al. (2012) principles using constituency parses.
  Comparison via Jensen-Shannon Divergence.
- Lexical-level: Four stylistic spectrums (subjective-objective, etc.) using NPMI
  with seed words. Comparison via MSE.
  Note: Does not implement the kNN graph/label propagation step from Syed et al. for word vectors.

Prompts user to select analyses to run.
"""
import math
import nltk
import re
import spacy # Assuming spaCy for tokenization if Benepar is used
import benepar # Ensure benepar is installed: pip install benepar
from nltk import Tree
from scipy.spatial.distance import jensenshannon
import logging
import os
import numpy as np # For JSD and other numerical ops

# Default file paths (User: Update these as needed)
FILE_1 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Mark_Twain/Splits/eval_every_15th_para_sample.txt"
FILE_2 = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Data/Charles_Dickens/Splits/eval_every_15th_para_sample.txt"

# --- Setup Logging ---
LOG_DIRECTORY = "/Users/finnsommer/LLaMa_FineTuning_UniCluster/llama-finetune/Style_Analysis/Output/Log_Files"
LOG_FILE_PATH = os.path.join(LOG_DIRECTORY, "analysis_constituency.log")
os.makedirs(LOG_DIRECTORY, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root_logger.addHandler(ch)
try:
    fh = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
except Exception as e:
    print(f"Error setting up file logger at {LOG_FILE_PATH}: {e}")
logger = logging.getLogger(__name__)

# --- Global Variables and Model Loading ---
nlp_spacy_benepar = None # spaCy model with Benepar pipe
PUNKT_AVAILABLE = False

# For normalized surface features (optional method)
max_diffs = {
    "avg_words_per_sentence": 10.0, "avg_commas_per_sentence": 2.0,
    "avg_semicolons_per_sentence": 0.5, "avg_colons_per_sentence": 0.3,
    "avg_sentences_per_paragraph": 3.0
}

# --- Seed Word Loading ---
try:
    from seed_loader import load_seed_words # Assuming seed_loader.py is available
    seed_lists = load_seed_words()
except ImportError:
    logger.warning("seed_loader.py not found or load_seed_words failed. Lexical analysis will use empty seed lists.")
    seed_lists = {}
except Exception as e:
    logger.error(f"Error loading seed words: {e}", exc_info=True)
    seed_lists = {}

LITERARY_SEEDS = set(seed_lists.get("literary", []))
ABSTRACT_SEEDS = set(seed_lists.get("abstract", []))
OBJECTIVE_SEEDS = set(seed_lists.get("objective", []))
COLLOQUIAL_SEEDS = set(seed_lists.get("colloquial", []))
CONCRETE_SEEDS = set(seed_lists.get("concrete", []))
SUBJECTIVE_SEEDS = set(seed_lists.get("subjective", []))
FORMAL_SEEDS = set(seed_lists.get("formal", []))
INFORMAL_SEEDS = set(seed_lists.get("informal", []))


# --- Helper Functions for Resource Checking ---
def ensure_nltk_punkt():
    global PUNKT_AVAILABLE
    if PUNKT_AVAILABLE: return True
    try:
        nltk.data.find('tokenizers/punkt'); PUNKT_AVAILABLE = True; logger.info("NLTK 'punkt' resource found.")
    except LookupError:
        logger.warning("NLTK 'punkt' not found. Attempting download...")
        try: nltk.download('punkt', quiet=True); nltk.data.find('tokenizers/punkt'); PUNKT_AVAILABLE = True; logger.info("NLTK 'punkt' downloaded.")
        except Exception as e: logger.error(f"Failed to download 'punkt': {e}. Manual download needed.", exc_info=True); PUNKT_AVAILABLE = False
    return PUNKT_AVAILABLE

def load_spacy_benepar_model():
    global nlp_spacy_benepar
    if nlp_spacy_benepar is None:
        logger.info("Loading spaCy model (en_core_web_sm) and Benepar...")
        try:
            nlp_spacy_benepar = spacy.load("en_core_web_sm")
            if "benepar" not in nlp_spacy_benepar.pipe_names:
                logger.info("Adding Benepar pipe to spaCy model...")
                nlp_spacy_benepar.add_pipe("benepar", config={"model": "benepar_en3"})
            logger.info("spaCy model and Benepar loaded.")
        except ImportError: logger.error("Benepar library not found. Install: pip install benepar"); raise
        except OSError as e: logger.error(f"Failed to load spaCy/Benepar models. Download them. Error: {e}"); raise
    return nlp_spacy_benepar

# ---------------------------
# Part 1: Surface-Level Features
# ---------------------------
def get_surface_features(text):
    # (Implementation remains largely the same as your provided version, ensuring NLTK punkt is checked)
    logger.info("Calculating surface features...")
    if not ensure_nltk_punkt():
        logger.error("NLTK 'punkt' unavailable for surface features.")
        return {k: 0.0 for k in max_diffs.keys()}

    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    if num_sentences == 0:
        logger.warning("No sentences found for surface features."); return {k: 0.0 for k in max_diffs.keys()}
    
    logger.info(f"Number of sentences: {num_sentences}")
    total_words = sum(len(nltk.word_tokenize(sent)) for sent in sentences)
    avg_words_per_sentence = total_words / num_sentences

    avg_commas_per_sentence = sum(sent.count(",") for sent in sentences) / num_sentences
    avg_semicolons_per_sentence = sum(sent.count(";") for sent in sentences) / num_sentences
    avg_colons_per_sentence = sum(sent.count(":") for sent in sentences) / num_sentences

    paragraphs = re.split(r'\n\s*\n', text.strip())
    num_paragraphs = len([p for p in paragraphs if p.strip()])
    avg_sentences_per_paragraph = sum(len(nltk.sent_tokenize(p.strip())) for p in paragraphs if p.strip()) / num_paragraphs if num_paragraphs > 0 else 0.0
    
    logger.info("Surface features calculation complete.")
    return {
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_commas_per_sentence": avg_commas_per_sentence,
        "avg_semicolons_per_sentence": avg_semicolons_per_sentence,
        "avg_colons_per_sentence": avg_colons_per_sentence,
        "avg_sentences_per_paragraph": avg_sentences_per_paragraph
    }

def compare_surface_features_mse(features1, features2):
    # This aligns with Syed et al. (2020) for surface features
    logger.info("Comparing surface features with MSE (Mean Squared Error).")
    keys = features1.keys() # Assuming both dicts have same keys
    squared_errors = [(features1.get(k, 0.0) - features2.get(k, 0.0))**2 for k in keys]
    mse = sum(squared_errors) / len(keys) if keys else 0.0
    similarity = 1.0 / (1.0 + mse) # Common conversion for MSE to similarity
    logger.info(f"Surface Features MSE: {mse:.6f}, Similarity (1/(1+MSE)): {similarity:.4f}")
    return mse, similarity

def compare_surface_features_normalized_abs_diff(features1, features2, p_max_diffs):
    # This is the alternative method the user wanted to keep as an option
    logger.info("Comparing surface features using Normalized Absolute Differences.")
    keys = features1.keys()
    normalized_abs_diffs = []
    for k in keys:
        if k in p_max_diffs and p_max_diffs[k] > 0:
            diff = abs(features1.get(k, 0.0) - features2.get(k, 0.0)) / p_max_diffs[k]
            normalized_abs_diffs.append(diff)
        else:
            logger.warning(f"Feature '{k}' skipped in normalized comparison (no valid max_diff).")
    
    if not normalized_abs_diffs: return float('inf'), 0.0
    avg_norm_abs_diff = sum(normalized_abs_diffs) / len(normalized_abs_diffs)
    similarity = 1.0 / (1.0 + avg_norm_abs_diff)
    logger.info(f"Avg Normalized Absolute Difference: {avg_norm_abs_diff:.6f}, Similarity: {similarity:.4f}")
    return avg_norm_abs_diff, similarity

# ---------------------------
# Part 2: Syntactic-Level Features (Feng et al., 2012 via Syed et al., 2020)
# ---------------------------
def _is_clause(tree_node, clausal_labels={'S', 'SINV', 'SQ'}):
    """ Helper to check if a tree node represents a clause. """
    return isinstance(tree_node, Tree) and tree_node.label() in clausal_labels

def _count_clauses_feng_et_al(tree):
    """
    Counts main and subordinate clauses in an NLTK Tree based on Feng et al. (2012)
    principles, adapted for typical constituency parse output.
    This is a heuristic implementation.
    """
    main_clauses = 0
    sub_clauses = 0

    # Simplified: Top-level S is often a main clause. SBAR always introduces subordinate.
    # More robustly: Traverse and identify clauses and their relationships.
    
    # Stack for DFS traversal of the tree
    nodes_to_visit = [tree]
    
    # Heuristic: identify main clauses
    # A simple sentence has one S. A compound sentence has multiple S joined by CC.
    # A complex sentence has an S and an SBAR.
    
    # Direct children of ROOT (or the main sentence node if no explicit ROOT)
    top_level_children = []
    if tree.label().upper() == 'ROOT' or tree.label().upper() == 'TOP': # Some parsers use ROOT or TOP
        top_level_children = [child for child in tree if isinstance(child, Tree)]
    elif _is_clause(tree): # If the tree itself is a clause node
         top_level_children = [tree]


    # Count top-level 'S' nodes not immediately dominated by 'SBAR' as potential main clauses
    potential_main_s_nodes = [ch for ch in top_level_children if ch.label() == 'S']
    
    if not potential_main_s_nodes and any(ch.label() == 'VP' for ch in top_level_children): # Imperative or fragment
        main_clauses = 1 # Treat as one main clause for simplicity
        
    elif len(potential_main_s_nodes) == 1:
        main_clauses = 1
    elif len(potential_main_s_nodes) > 1:
        # Check if they are coordinated (e.g., S CC S)
        is_compound = False
        for i in range(len(top_level_children) - 2):
            if top_level_children[i].label() == 'S' and \
               top_level_children[i+1].label() == 'CC' and \
               top_level_children[i+2].label() == 'S':
                is_compound = True
                break
        if is_compound:
            main_clauses = potential_main_s_nodes.count('S') # Count all S at top if coordinated
        else: # Multiple S nodes not clearly coordinated at top level might be nested or errors
            main_clauses = 1 # Default to 1, subordination handled by SBAR count

    # Count SBARs for subordinate clauses anywhere in the tree
    for subtree in tree.subtrees():
        if isinstance(subtree, Tree) and subtree.label() == 'SBAR':
            sub_clauses += 1
        # Also, an 'S' that is a child of VP or another S (and not an SBAR's child) can be a subordinate clause
        # This part is tricky and highly dependent on specific grammar and parser output.
        # Feng et al. might have more specific rules related to Stanford Parser output.
        # For now, SBAR is the primary indicator.

    # If no SBARs found, but it's a single S that seems to contain other clauses (e.g. S -> NP VP S), that's complex
    # This heuristic is simplified. A full implementation of Feng et al.'s clause counting is complex.
    if main_clauses == 0 and sub_clauses == 0 and _is_clause(tree): # If tree itself is clause, no sub, assume 1 main
        main_clauses = 1
        
    return main_clauses, sub_clauses


def classify_sentence_feng_et_al(sentence_text):
    nlp_instance = load_spacy_benepar_model()
    cleaned_sentence = sentence_text.strip() # Basic cleaning
    if not cleaned_sentence: return "other"

    try:
        doc = nlp_instance(cleaned_sentence)
        if not list(doc.sents): return "other"
        tree_str = list(doc.sents)[0]._.parse_string
        if not tree_str: return "other"
        tree = Tree.fromstring(tree_str)
    except Exception as e:
        logger.error(f"Error parsing sentence for Feng et al. classification '{cleaned_sentence[:100]}...': {e}")
        return "other"

    # Basic cleaning of the tree from common parser noise for this specific task if needed
    # e.g., removing punctuation-only subtrees if they interfere with clause counting

    num_main_clauses, num_sub_clauses = _count_clauses_feng_et_al(tree)

    if num_main_clauses == 1 and num_sub_clauses == 0:
        return "simple"
    elif num_main_clauses >= 2 and num_sub_clauses == 0:
        return "compound"
    elif num_main_clauses == 1 and num_sub_clauses >= 1:
        return "complex"
    elif num_main_clauses >= 2 and num_sub_clauses >= 1:
        return "complex-compound"
    else:
        # This 'other' could also catch cases where clause counting returned (0,0) for a non-empty parse
        logger.debug(f"Sentence '{cleaned_sentence[:50]}...' classified 'other' by Feng. Main: {num_main_clauses}, Sub: {num_sub_clauses}")
        return "other"

def get_syntactic_distribution(text):
    logger.info("Calculating syntactic distribution (Feng et al. categories)...")
    if not ensure_nltk_punkt(): logger.error("NLTK 'punkt' unavailable."); return [0.0]*5
    
    sentences = nltk.sent_tokenize(text)
    if not sentences: logger.warning("No sentences for syntactic dist."); return [0.0]*5
    logger.info(f"Found {len(sentences)} sentences for syntactic analysis.")

    counts = {"simple": 0, "compound": 0, "complex": 0, "complex-compound": 0, "other": 0}
    processed_count = 0
    for i, sent in enumerate(sentences):
        clean_sent = re.sub(r'\s+', ' ', sent).strip() # Minimal clean for classification
        if not clean_sent: continue
        
        cat = classify_sentence_feng_et_al(clean_sent)
        counts[cat] += 1
        processed_count += 1
        if (i + 1) % 100 == 0: logger.info(f"Syntactic classification: {i+1}/{len(sentences)} sentences processed.")
    
    if processed_count == 0: logger.warning("No sentences classified."); return [0.0]*5

    dist_order = ["simple", "compound", "complex", "complex-compound", "other"]
    distribution = [counts[cat] / processed_count for cat in dist_order]
    logger.info(f"Syntactic distribution: { {k: f'{v:.3f}' for k,v in zip(dist_order, distribution)} }")
    return distribution

def compute_syntactic_jsd(dist1, dist2):
    # (Implementation remains the same as your provided version, ensuring proper normalization)
    logger.info("Computing JSD for syntactic distributions.")
    d1_np, d2_np = np.array(dist1, dtype=float), np.array(dist2, dtype=float)
    eps_norm = 1e-9
    for d_arr in [d1_np, d2_np]: # In-place modification not ideal, but for this scope
        if not np.isclose(np.sum(d_arr), 1.0) and np.sum(d_arr) > 0:
            d_arr[:] = (d_arr + eps_norm) / np.sum(d_arr + eps_norm * len(d_arr)) # Modify in place
    if np.sum(d1_np) == 0 or np.sum(d2_np) == 0:
        logger.warning("Zero distribution in JSD calc."); return 1.0 if not np.array_equal(d1_np, d2_np) else 0.0
    try:
        jsd = jensenshannon(d1_np, d2_np, base=2.0); logger.info(f"JSD: {jsd:.6f}"); return jsd
    except ValueError as ve: logger.error(f"JSD ValueError: {ve}. D1: {d1_np}, D2: {d2_np}"); return 1.0

# ---------------------------
# Part 3 & 4: Lexical Analysis (NPMI as per Syed et al., without kNN/label propagation)
# ---------------------------
def get_lexical_style_full_syed_et_al_inspired(text):
    # This function aims to follow Syed et al.'s NPMI approach for word association
    # BUT DOES NOT IMPLEMENT the kNN graph and label propagation step for word vectors.
    # It computes average spectrum scores directly from aggregated NPMI.
    logger.info("Calculating lexical style (Syed et al. inspired NPMI, no kNN/label prop)...")
    # (Implementation from your provided script, with `add_constant` removed and NPMI shifted for spectrum, is kept)
    # (Ensure ensure_nltk_punkt() is called here as well)
    if not ensure_nltk_punkt():
        logger.error("NLTK 'punkt' unavailable for full lexical analysis.")
        return {"subjective_obj": 0.0, "concrete_abstract": 0.0, "literary_colloquial": 0.0, "formal_informal": 0.0}

    eps = 1e-9
    sentences = nltk.sent_tokenize(text)
    docs = [set(nltk.word_tokenize(sent.lower())) for sent in sentences if sent.strip()]
    docs = [doc for doc in docs if doc]
    total_docs = len(docs)

    if total_docs == 0:
        logger.warning("No documents for full lexical analysis."); return {k:0.0 for k in ["subjective_obj", "concrete_abstract", "literary_colloquial", "formal_informal"]}

    vocabulary = set.union(*docs) if docs else set()
    if not vocabulary:
        logger.warning("Empty vocabulary for lexical analysis."); return {k:0.0 for k in ["subjective_obj", "concrete_abstract", "literary_colloquial", "formal_informal"]}

    df = {w: sum(1 for doc_set in docs if w in doc_set) for w in vocabulary}
    seed_categories_map = {
        "subj": SUBJECTIVE_SEEDS, "obj": OBJECTIVE_SEEDS, "conc": CONCRETE_SEEDS,
        "abst": ABSTRACT_SEEDS, "lit": LITERARY_SEEDS, "coll": COLLOQUIAL_SEEDS,
        "form": FORMAL_SEEDS, "infm": INFORMAL_SEEDS
    }
    active_seed_words_by_cat = {
        cat_name: set(w.lower() for w in s_list if w.lower() in vocabulary)
        for cat_name, s_list in seed_categories_map.items()
    }
    p_seed = {s_word: (df.get(s_word, 0) + eps) / (total_docs + eps)
              for s_set in active_seed_words_by_cat.values() for s_word in s_set}
    p_seed = {k:v for k,v in p_seed.items()} # Ensure all seeds considered

    def calculate_npmi(p_xy, p_x, p_y, smoothing_eps=1e-12):
        p_xy, p_x, p_y = max(p_xy, smoothing_eps), max(p_x, smoothing_eps), max(p_y, smoothing_eps)
        try: pmi_val = math.log(p_xy / (p_x * p_y), 2)
        except ValueError: pmi_val = -float('inf')
        log_p_xy = math.log(p_xy, 2)
        if p_xy == 1.0 or log_p_xy == 0 : return 1.0 if pmi_val >=0 else -1.0
        return max(-1.0, min(1.0, pmi_val / (-log_p_xy)))

    raw_word_scores = {w: {cat_name: 0.0 for cat_name in seed_categories_map} for w in vocabulary}
    for w_idx, w in enumerate(vocabulary):
        if (w_idx + 1) % 500 == 0: logger.info(f"Lexical NPMI: word {w_idx+1}/{len(vocabulary)}")
        p_w = (df.get(w, 0) + eps) / (total_docs + eps)
        if p_w <= eps / total_docs: continue

        for cat_name, current_cat_seed_set in active_seed_words_by_cat.items():
            cat_npmi_sum, num_seeds = 0.0, 0
            for s_word in current_cat_seed_set:
                if s_word == w: continue
                co_occurrence_count = sum(1 for doc_set in docs if w in doc_set and s_word in doc_set)
                p_ws = (co_occurrence_count + eps) / (total_docs + eps)
                current_p_s = p_seed.get(s_word, eps / total_docs)
                cat_npmi_sum += calculate_npmi(p_ws, p_w, current_p_s)
                num_seeds += 1
            if num_seeds > 0: raw_word_scores[w][cat_name] = cat_npmi_sum / num_seeds
    
    # Create 4 spectrums by averaging word scores (as per Syed et al. before comparison)
    # "The averages, in the range [0, 1], denote the tendency..."
    # This implies the spectrums are formed for the whole text, not per word then averaged.
    # "we compute 4 averages across the entire author-specific corpus" - this could mean
    # averaging the word scores first for each of the 8 categories, then forming spectrums.
    
    # Let's follow: "average of these spectrum scores over all words in a document is used as a 4-D vector"
    # This means: 1. Word -> 8 scores. 2. Word -> 4 spectrum scores. 3. Doc -> average of 4 spectrum scores.
    
    doc_spectrum_accumulator = {"subjective_obj":0.0, "concrete_abstract":0.0, "literary_colloquial":0.0, "formal_informal":0.0}
    valid_words_for_spectrum = 0

    for w in vocabulary:
        # Shift NPMI scores [-1, 1] to [0, 2] for ratio calculation
        s_subj = raw_word_scores[w]["subj"] + 1.0
        s_obj  = raw_word_scores[w]["obj"]  + 1.0
        s_conc = raw_word_scores[w]["conc"] + 1.0
        s_abst = raw_word_scores[w]["abst"] + 1.0
        s_lit  = raw_word_scores[w]["lit"]  + 1.0
        s_coll = raw_word_scores[w]["coll"] + 1.0
        s_form = raw_word_scores[w]["form"] + 1.0
        s_infm = raw_word_scores[w]["infm"] + 1.0

        doc_spectrum_accumulator["subjective_obj"]     += s_subj / (s_subj + s_obj + eps)
        doc_spectrum_accumulator["concrete_abstract"]  += s_conc / (s_conc + s_abst + eps)
        doc_spectrum_accumulator["literary_colloquial"]+= s_lit  / (s_lit  + s_coll + eps)
        doc_spectrum_accumulator["formal_informal"]    += s_form / (s_form + s_infm + eps)
        valid_words_for_spectrum +=1
        
    if valid_words_for_spectrum == 0:
        logger.warning("No words processed for spectrum calculation in full lexical analysis."); return {k:0.0 for k in doc_spectrum_accumulator}

    final_text_lexical_vector = { k: v / valid_words_for_spectrum for k, v in doc_spectrum_accumulator.items() }
    logger.info(f"Full lexical style vector (Syed-inspired): {final_text_lexical_vector}")
    return final_text_lexical_vector

def compare_lexical_vectors_mse(vec1, vec2):
    # This aligns with Syed et al. (2020) for lexical features
    return compare_surface_features_mse(vec1, vec2) # MSE calculation is generic

# Simple seed count (kept as an alternative if user chooses)
def get_lexical_style_simple(text):
    # (Implementation remains the same as your provided version, ensuring NLTK punkt is checked)
    logger.info("Calculating simple lexical style (seed counts)...")
    if not ensure_nltk_punkt():
        logger.error("NLTK 'punkt' unavailable for simple lexical analysis.")
        return {k:0.0 for k in ["subjective_obj", "concrete_abstract", "literary_colloquial", "formal_informal"]}
    
    tokens = nltk.word_tokenize(text.lower())
    if not tokens: logger.warning("No tokens for simple lexical analysis."); return {k:0.0 for k in ["subjective_obj", "concrete_abstract", "literary_colloquial", "formal_informal"]}

    counts = {cat: sum(1 for t in tokens if t in s_list) for cat, s_list in {
        "subj": SUBJECTIVE_SEEDS, "obj": OBJECTIVE_SEEDS, "conc": CONCRETE_SEEDS,
        "abst": ABSTRACT_SEEDS, "lit": LITERARY_SEEDS, "coll": COLLOQUIAL_SEEDS,
        "form": FORMAL_SEEDS, "infm": INFORMAL_SEEDS}.items()}
    eps = 1e-9
    vec = {
        "subjective_obj": counts["subj"] / (counts["subj"] + counts["obj"] + eps),
        "concrete_abstract": counts["conc"] / (counts["conc"] + counts["abst"] + eps),
        "literary_colloquial": counts["lit"] / (counts["lit"] + counts["coll"] + eps),
        "formal_informal": counts["form"] / (counts["form"] + counts["infm"] + eps)
    }
    logger.info(f"Simple lexical vector: {vec}")
    return vec

# ---------------------------
# Main Program
# ---------------------------
def main():
    logger.info(f"--- Starting Analysis Script ---")
    # Ensure NLTK data is available at the beginning if any NLTK dependent function will be called
    ensure_nltk_punkt()

    logger.info(f"Reading File 1: {FILE_1}")
    try: text1 = open(FILE_1, "r", encoding="utf-8").read()
    except Exception as e: logger.error(f"Failed to read {FILE_1}: {e}", exc_info=True); return
    logger.info(f"Reading File 2: {FILE_2}")
    try: text2 = open(FILE_2, "r", encoding="utf-8").read()
    except Exception as e: logger.error(f"Failed to read {FILE_2}: {e}", exc_info=True); return

    # --- Surface-Level Analysis ---
    ans_surf = input("Perform surface-level analysis? [y|n]: ").strip().lower()
    if ans_surf.startswith("y"):
        logger.info("User requested surface-level analysis.")
        surf_method = input("  Compare surface features using (1) MSE (Syed et al.) or (2) Normalized Absolute Difference? [1|2]: ").strip()
        
        s1 = get_surface_features(text1)
        s2 = get_surface_features(text2)
        print("\nSurface Features (File 1):", {k: f"{v:.3f}" for k,v in s1.items()})
        print("Surface Features (File 2):", {k: f"{v:.3f}" for k,v in s2.items()})

        if surf_method == '2':
            val, sim = compare_surface_features_normalized_abs_diff(s1, s2, max_diffs)
            print(f"\nSurface-Level Avg Normalized Absolute Difference: {val:.6f}, Similarity: {sim:.4f}")
        else: # Default to MSE (Syed et al.)
            val, sim = compare_surface_features_mse(s1, s2)
            print(f"\nSurface-Level MSE: {val:.6f}, Similarity (1/(1+MSE)): {sim:.4f}")
    else:
        logger.info("Skipping surface-level analysis.")

    # --- Syntactic-Level Analysis ---
    ans_synt = input("\nPerform syntactic-level analysis (Feng et al. categories)? [y|n]: ").strip().lower()
    if ans_synt.startswith("y"):
        logger.info("User requested syntactic-level analysis.")
        try:
            load_spacy_benepar_model() # Ensure models are ready
            dist1 = get_syntactic_distribution(text1)
            dist2 = get_syntactic_distribution(text2)
            dist_labels = ["Simple", "Compound", "Complex", "Cx-Comp", "Other"]
            print("\nSyntactic Distribution (File 1):", {k: f"{v:.3f}" for k,v in zip(dist_labels, dist1)})
            print("Syntactic Distribution (File 2):", {k: f"{v:.3f}" for k,v in zip(dist_labels, dist2)})
            jsd = compute_syntactic_jsd(dist1, dist2)
            print(f"\nSyntactic JSD: {jsd:.4f} (Lower is more similar)")
            print(f"Syntactic Similarity (1 - JSD): {1.0 - jsd:.4f}")
        except Exception as e: logger.error(f"Error during syntactic analysis: {e}", exc_info=True)
    else:
        logger.info("Skipping syntactic-level analysis.")

    # --- Lexical Analysis ---
    ans_lex = input("\nPerform lexical analysis? [y|n]: ").strip().lower()
    if ans_lex.startswith("y"):
        logger.info("User requested lexical analysis.")
        if not any(s for s_list in seed_lists.values() for s in s_list):
             logger.warning("Seed lists appear globally empty based on initial load. Lexical analysis results will be zero.")

        lex_type = input("  Choose lexical analysis: (1) Simple Seed Count or (2) NPMI-based (Syed et al. inspired)? [1|2]: ").strip()
        if lex_type == '1':
            logger.info("Performing SIMPLE lexical analysis.")
            v1, v2 = get_lexical_style_simple(text1), get_lexical_style_simple(text2)
        else: # Default to NPMI
            logger.info("Performing FULL lexical analysis (Syed et al. inspired NPMI).")
            logger.warning("Note: This implementation does NOT include the kNN graph/label propagation step mentioned in Syed et al. (2020) for word vectors due to its complexity.")
            v1, v2 = get_lexical_style_full_syed_et_al_inspired(text1), get_lexical_style_full_syed_et_al_inspired(text2)
        
        print("\nLexical Style Vector (File 1):", {k: f"{v_val:.3f}" for k,v_val in v1.items()})
        print("Lexical Style Vector (File 2):", {k: f"{v_val:.3f}" for k,v_val in v2.items()})
        mse, sim = compare_lexical_vectors_mse(v1, v2) # Syed et al. use MSE for lexical
        print(f"\nLexical Style MSE: {mse:.6f}, Similarity (1/(1+MSE)): {sim:.4f}")
    else:
        logger.info("Skipping lexical analysis.")

    logger.info("--- Analysis script finished. ---")

if __name__ == "__main__":
    main()