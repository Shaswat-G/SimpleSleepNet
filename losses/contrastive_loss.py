import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def normalize_embeddings(embedding_1, embedding_2):
    """
    Normalizes the embeddings along the specified dimension.

    Parameters:
    - embedding_1, embedding_2: Embeddings to be normalized (batch_size, embedding_dim).

    Returns:
    - Normalized embeddings.
    """
    embedding_1 = nn.functional.normalize(embedding_1, dim=1)
    embedding_2 = nn.functional.normalize(embedding_2, dim=1)
    return embedding_1, embedding_2

def concatenate_embeddings(embedding_1, embedding_2):
    """
    Concatenates two embeddings along the batch dimension.

    Parameters:
    - embedding_1, embedding_2: Embeddings to be concatenated (batch_size, embedding_dim).

    Returns:
    - Concatenated embeddings (2*batch_size, embedding_dim).
    """
    return torch.cat([embedding_1, embedding_2], dim=0)

def compute_similarity_matrix(embeddings, temperature):
    """
    Computes the similarity matrix for the embeddings.

    Parameters:
    - embeddings: Concatenated embeddings (2*batch_size, embedding_dim).
    - temperature: Temperature scaling factor.

    Returns:
    - Similarity matrix (2*batch_size, 2*batch_size).
    """
    return torch.matmul(embeddings, embeddings.T) / temperature

def mask_self_similarities(similarity_matrix, batch_size, device):
    """
    Masks the self-similarities in the similarity matrix.

    Parameters:
    - similarity_matrix: Similarity matrix (2*batch_size, 2*batch_size).
    - batch_size: Batch size of the embeddings.
    - device: Device on which the tensors are located.

    Returns:
    - Similarity matrix with self-similarities masked.
    """
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
    similarity_matrix.masked_fill_(mask, -float('inf'))
    return similarity_matrix

def nt_xent_loss(embedding_1, embedding_2, temperature=0.5):
    """
    Computes the NT-Xent loss as introduced in SimCLR.

    Parameters:
    - embedding_1, embedding_2: Normalized embeddings of two augmented views (batch_size, embedding_dim).
    - temperature: Temperature scaling factor.

    Returns:
    - loss: The computed NT-Xent loss.
    """
    batch_size = embedding_1.size(0)
    device = embedding_1.device

    # Normalize embeddings
    embedding_1, embedding_2 = normalize_embeddings(embedding_1, embedding_2)
    # Concatenate embeddings
    embeddings = concatenate_embeddings(embedding_1, embedding_2)
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings, temperature)
    # Remove self-similarities
    similarity_matrix = mask_self_similarities(similarity_matrix, batch_size, device)

    # Positive sample indices
    labels = torch.arange(batch_size).to(device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    # Compute loss
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

    return loss