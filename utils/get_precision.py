def get_precision(labels, outputs, num_clases):
    precisions = []
    for i in range(num_clases+1):
        labels = [a for a in labels if i == a else -1]
        (labels == num_clases).sum()
