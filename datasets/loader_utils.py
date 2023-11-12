import random
from itertools import combinations

def shuffle_dataset(features, bag_label, random_seed=42, instance_labels=None):

    """
    Random permutation of the dataset
    :param features: [list]
    :param bag_label: [list]
    :param random_seed: [int]]
    :return: shuffled dataset
    """
    if instance_labels:
        index = list(zip(features, bag_label, instance_labels))
    else:
        index = list(zip(features, bag_label))
    random.seed(random_seed)
    random.shuffle(index)
    if instance_labels:
        features, bag_label, instance_labels = zip(*index)
        return list(features), list(bag_label), list(instance_labels)
    else:
        features, bag_label = zip(*index)
        return list(features), list(bag_label)

def multiply_features(list_of_instance):
    length = len(list_of_instance[0])
    all_combinations = list(combinations([a for a in range(length)],2))
    for el1, el2 in all_combinations:
        list_of_instance = np.c_[list_of_instance, list_of_instance[:,el1][:,] * list_of_instance[:,el2][:,]]
    return list_of_instance

def train_test_split(features, bag_labels, instance_labels=None):
    
    # ToDos
    split = 0.9
    # Check split size    

    num_bags = len(features)
    index = int(split * num_bags)
    if instance_labels:
        return features[:index], features[index:], bag_labels[:index], bag_labels[index:], instance_labels[:index], instance_labels[index:]
    else:
        return features[:index], features[index:], bag_labels[:index], bag_labels[index:]

def into_dictionary(index, features):
    """
    helper function for transforming dataset from list to dict
    :param index: index of the instance [list]
    :param features: features  [list]
    :return: dictionary [dict]
    """
    dictionary = {}
    for index, value in zip(index, features):
        if index in dictionary:
            dictionary[index].append(value)
        else:
            dictionary[index] = [value]
    return dictionary