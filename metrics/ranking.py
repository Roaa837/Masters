from scipy.stats import spearmanr, kendalltau

def compute_ranking(pred_scores, ref_scores):
    spearman = spearmanr(pred_scores, ref_scores).correlation
    kendall = kendalltau(pred_scores, ref_scores).correlation

    return {
        "spearman": spearman,
        "kendall": kendall
    }