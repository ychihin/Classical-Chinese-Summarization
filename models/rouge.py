from collections import Counter

def rouge_1 (orig, tran):
    orig_count = Counter(list(orig))
    tran_count = Counter(list(tran))
    combine = orig_count & tran_count
    overlap_count = sum(combine.values())
    precision = overlap_count / len(tran)
    recall = overlap_count / len(orig)
    F1 = (2 * precision * recall) / (precision + recall)
    return (precision, recall, F1)

def rouge_2 (orig, tran):
    orig_bigram = [orig[i:i+2] for i in range(len(orig)-1)]
    tran_bigram = [tran[i:i+2] for i in range(len(tran)-1)]
    orig_count = Counter(list(orig_bigram))
    tran_count = Counter(list(tran_bigram))
    combine = orig_count & tran_count
    overlap_count = sum(combine.values())
    recall = overlap_count / len(orig_bigram)
    precision = overlap_count / len(tran_bigram)
    F1 = (2 * precision * recall) / (precision + recall)
    return (precision, recall, F1)

# Calculate largest common sequence used in ROGUE-L
def lcs_length(x, y):
    m, n = len(x), len(y)
    grid = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                grid[i][j] = grid[i - 1][j - 1] + 1
            else:
                grid[i][j] = max(grid[i - 1][j], grid[i][j - 1])

    return grid[m][n]

def rouge_l (orig, tran):
    lcs = lcs_length(orig, tran)
    precision = lcs / len(tran)
    recall = lcs / len(orig)
    F1 = (2 * precision * recall) / (precision + recall)
    return (precision, recall, F1)

