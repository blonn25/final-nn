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
    # separate the sequences into positive and negative classes
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    # apply random oversampling to the minority class
    if len(pos_seqs) < len(neg_seqs):
        pos_seqs = np.random.choice(pos_seqs, size=len(neg_seqs), replace=True).tolist()
    elif len(neg_seqs) < len(pos_seqs):
        neg_seqs = np.random.choice(neg_seqs, size=len(pos_seqs), replace=True).tolist()
    
    # recombine the oversampled sequences and labels
    sampled_seqs = pos_seqs + neg_seqs
    sampled_labels = [True] * len(pos_seqs) + [False] * len(neg_seqs)

    # return the sampled sequences and labels
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
    # create a mapping from nucleotides to one-hot encodings
    nt_to_onehot = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1]
    }

    # convert to seqs to a list of one-hot encodings (stored as a list of lists)
    encodings = []
    for seq in seq_arr:
        seq_encoding = []
        for nt in seq:
            seq_encoding.extend(nt_to_onehot[nt])
        encodings.append(seq_encoding)

    # return the flattened sequence encodings
    return encodings