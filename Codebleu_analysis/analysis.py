import sys
sys.path.append('/mnt/d/MyFile/project/LLM_code_review/code/framework/Codebleu_analysis')
from codebleu.codebleu.codebleu import calc_codebleu

def calculate_codebleu_score(reference,prediction,score_type='codebleu'):
    result = calc_codebleu([reference], [prediction], lang="python", weights=(0.3, 0.4, 0, 0.3), tokenizer=None)
    return result
# {
#   'codebleu': 0.5537, 
#   'ngram_match_score': 0.1041, 
#   'weighted_ngram_match_score': 0.1109, 
#   'syntax_match_score': 1.0, 
#   'dataflow_match_score': 1.0
# }