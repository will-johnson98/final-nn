# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Convert labels to numpy array for easier manipulation
    labels_array = np.array(labels)
    positive_indices = np.where(labels_array == True)[0]
    negative_indices = np.where(labels_array == False)[0]
    
    num_positive = len(positive_indices)
    num_negative = len(negative_indices)
    max_class_size = max(num_positive, num_negative)
    
    if num_positive < num_negative:
        sampled_positive_indices = np.random.choice(positive_indices, max_class_size, replace=True)
        sampled_negative_indices = negative_indices
    else:
        sampled_positive_indices = positive_indices
        sampled_negative_indices = np.random.choice(negative_indices, max_class_size, replace=True)
    
    sampled_indices = np.concatenate([sampled_positive_indices, sampled_negative_indices])
    np.random.shuffle(sampled_indices)
    sampled_seqs = [seqs[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Define the encoding dictionary
    encoding_dict = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'a': [1, 0, 0, 0],
        't': [0, 1, 0, 0],
        'c': [0, 0, 1, 0],
        'g': [0, 0, 0, 1],
        'N': [0, 0, 0, 0],
        'n': [0, 0, 0, 0]
    }
    all_encodings = []
    
    for seq in seq_arr:
        encoding = []
        
        for nucleotide in seq:
            nucleotide_encoding = encoding_dict.get(nucleotide, [0, 0, 0, 0])
            encoding.extend(nucleotide_encoding)
        all_encodings.append(encoding)
    
    return np.array(all_encodings)
