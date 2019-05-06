"""
Some helper scripts to quantitatively compute the performance
of the model
"""
def computeF1(goldList, predictedList):
    '''
    return a tuple with recall, precision, and f1 for one example. This
    scripted was originally implemented with the assumption that the questions
    might have multiple solutions. With small fix, this could be adapt to
    single-solution case as well.
    Adapted from https://github.com/ysu1989/GraphQuestions/blob/255ebc92301f93afd5b98165e431833a3cba38e7/evaluate.py
    '''

    # convert the prediction and ground truth label to list
    # in the case of single-solution question
    if not isinstance(goldList, list):
        goldList = [goldList]
    if not isinstance(predictedList, list):
        predictedList = [predictedList]

    # Assume questions have at least one answer
    if len(goldList) == 0:
        raise Exception('gold list may not be empty')
    # If we return an empty list recall is zero and precision is one
    if len(predictedList) == 0:
        return (0, 1, 0)
    # It is guaranteed now that neither of the lists is empty

    precision = 0
    for entity in predictedList:
        if entity in goldList:
            precision += 1
    precision = float(precision) / len(predictedList)

    recall = 0
    for entity in goldList:
        if entity in predictedList:
            recall += 1
    recall = float(recall) / len(goldList)

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * recall * precision / (precision + recall)
    return (recall, precision, f1)
