import numpy as np


###############################################
################ METRICS ######################
###############################################

def precision(is_relevant):

    """
    Compute the precision metric TRUE_POSITIVES/#ITEMS_TO_RECOMMEND
    :param is_relevant: booleans array, true means that the item is relevant
    :return: precision value
    """

    true_positives = is_relevant.sum(dtype=np.float32)
    return true_positives / len(is_relevant)


def recall(is_relevant, relevant_items):

    """
    Compute the recall metric: TRUE_POSITIVES / TRUE_POS+FALSE_NEGATIVES
    :param is_relevant: booleans array, true means that the item is relevant
    :param relevant_items: items that have been already interacted with
    :return: recall value
    """

    true_positives = is_relevant.sum(dtype=np.float32)
    if relevant_items.shape[0] == 0:
        return 0
    return true_positives / relevant_items.shape[0]


def map_at(is_relevant, relevant_items):

    """
    Compute the Mean Average Precision
    :param is_relevant: booleans array, true means that the item is relevant
    :param relevant_items: items that have been already interacted with
    :return: MAP value
    """

    precision_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    minimum = np.min([relevant_items.shape[0], is_relevant.shape[0]])

    if minimum == 0:
        return 0
    return np.sum(precision_at_k) / minimum