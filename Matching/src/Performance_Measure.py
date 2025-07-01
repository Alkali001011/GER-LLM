from Blocking.src.tools.utils import load_pkl


def measure_performance(all_batches, all_answers, labels_address):
    set_gt = load_pkl(labels_address)

    n = len(all_batches)

    tp = 0  # 真正类：实际匹配，且正确被标为1的
    for i in range(0, n):
        pair = all_batches[i]
        se1, se2 = pair
        pair_tuple = (se1.id, se2.id)

        if all_answers[i] == 1:  # 如果预测结果为1
            if pair_tuple in set_gt:  # 如果实际标签也为1
                tp = tp + 1

    fn = len(set_gt) - tp  # 假负类：实际匹配，但被错标为0的
    fp = all_answers.count(1) - tp  # 假正类：实际不匹配，但被错标成1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1-score: ' + str(f1))
