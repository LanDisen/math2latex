import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
import math

# Metrics: 
# BLEU Score, Edit Distance Score, Exact Match Score, overall scores
def bleu_score(references, hypotheses):
    '''BLEU Score (0~100; larger is better)'''
    score = 0.0 
    for i, j in zip(references, hypotheses): 
        score += max(sentence_bleu([i], j), 0.01) 
    score = score / len(references) * 100
    return score

def edit_distence(references, hypotheses):
    """
    Edit Distance Score (0~100; larger is better)
    Computes Levenshtein distance between two sequences.
    Args:
        references: list of sentences (one hypothesis)[真实答案]
        hypotheses: list of sentences (one hypothesis)[模型预测的答案]
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        # d_leven += distance.levenshtein(ref, hypo)
        d_leven += Levenshtein.distance(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))
    
    return (1. - d_leven / len_tot) * 100

def exact_match_score(references, hypotheses):
    '''
    Exact Match Score (0~100; larger is better)
    每个样本完全匹配才算正确
    Exact Match Score = Accuracy * 100
    '''
    total_num = len(references)
    acc_num = 0
    for ref, hypo in zip(references, hypotheses):
        if ref == hypo:
            acc_num += 1
    return (acc_num / total_num) * 100

def overall_score(references, hypotheses):
    '''
    总体分数, 指标1, 2, 3的均值, 最后排名的依据
    '''
    score1 = bleu_score(references, hypotheses)
    score2 = edit_distence(references, hypotheses)
    score3 = exact_match_score(references, hypotheses)
    return (score1 + score2 + score3) / 3
