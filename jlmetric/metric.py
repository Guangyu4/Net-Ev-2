import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import TemporalData
from typing import Dict, List
import numpy as np
from tqdm import tqdm

class JLEvaluator:
    """
    Evaluator for Continuous-Time Dynamic Graphs (CTDGs) using the JL-Metric.

    The JL-Metric compares generated CTDGs against reference graphs by projecting
    node and graph representations into lower-dimensional spaces and measuring similarity.

    Args:
        node_dim (int, optional): Dimension for node embeddings. Default: 100
        graph_dim (int, optional): Dimension for graph embeddings. Default: 100
        seed (int, optional): Random seed for reproducibility. Default: 42
    """

    def __init__(self, node_dim: int = 100, graph_dim: int = 100, seed: int = 42, device: str = 'cuda'):
        self.node_dim = node_dim
        self.graph_dim = graph_dim
        self.seed = seed
        self.MAX_EVENTS = 100000
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def eval(self, input_dict: Dict[str, TemporalData]) -> Dict[str, float]:
        """
        Evaluate the similarity between a generated CTDG and a reference CTDG.

        Args:
            input_dict (Dict[str, TemporalData]): Dictionary containing:
                - 'reference': Reference CTDG as PyG TemporalData
                - 'generated': Generated CTDG as PyG TemporalData

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'JL-Metric': Similarity score (higher is better)
        """
        reference_graph = input_dict["reference"]
        generated_graph = input_dict["generated"]

        # Extract graph representations
        reference_embedding = self._compute_graph_embedding(reference_graph)
        generated_embedding = self._compute_graph_embedding(generated_graph)

        # Compute cosine similarity between graph embeddings
        similarity = self._compute_similarity(reference_embedding, generated_embedding)

        return {"JL-Metric": similarity}

    def batch_eval(self, reference_graphs: List[TemporalData], generated_graphs: List[TemporalData]) -> Dict[str, float]:
        if len(reference_graphs) != len(generated_graphs):
            raise ValueError(f"Number of reference graphs ({len(reference_graphs)}) must match "
                           f"number of generated graphs ({len(generated_graphs)})")
        
        if len(reference_graphs) == 0:
            raise ValueError("Input lists cannot be empty")
        
        reference_embeddings = self._batch_compute_graph_embeddings(reference_graphs)
        generated_embeddings = self._batch_compute_graph_embeddings(generated_graphs)
        
        similarities = self._batch_compute_similarities(reference_embeddings, generated_embeddings)
        
        return reference_embeddings, generated_embeddings, similarities

    def _batch_compute_graph_embeddings(self, graphs: List[TemporalData]) -> Tensor:
        torch_geometric.seed_everything(self.seed)
        
        batch_size = len(graphs)
        embd = torch.randn((self.MAX_EVENTS, self.node_dim), device=self.device) * torch.sqrt(
            torch.tensor(1.0 / self.MAX_EVENTS, device=self.device)
        )
        
        return batch_jl_metric(graphs, self.node_dim, self.graph_dim, self.seed, self.MAX_EVENTS, self.device)

    def _batch_compute_similarities(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        norm1 = torch.norm(embeddings1, dim=-1, keepdim=True)
        norm2 = torch.norm(embeddings2, dim=-1, keepdim=True)
        
        embeddings1_normalized = embeddings1 / (norm1 + 1e-8)
        embeddings2_normalized = embeddings2 / (norm2 + 1e-8)
        
        similarities = torch.sum(embeddings1_normalized * embeddings2_normalized, dim=-1)
        similarities = torch.mean(similarities, dim=-1)
        
        return similarities

    def _compute_graph_embedding(self, events: TemporalData) -> Tensor:
        """
        Compute the graph-level embedding for a CTDG using JL projection.

        Args:
            events (TemporalData): PyG TemporalData object containing the CTDG

        Returns:
            Tensor: Graph-level embedding
        """
        return jl_metric(events, self.node_dim, self.graph_dim, self.seed)

    def _compute_similarity(self, embedding1: Tensor, embedding2: Tensor) -> float:
        """
        Compute the similarity between two graph embeddings.

        Args:
            embedding1 (Tensor): First graph embedding
            embedding2 (Tensor): Second graph embedding

        Returns:
            float: Similarity score (higher is better)
        """
        # Normalize embeddings
        norm1 = torch.norm(embedding1, dim=0, keepdim=True)
        norm2 = torch.norm(embedding2, dim=0, keepdim=True)

        embedding1_normalized = embedding1 / (norm1 + 1e-8)
        embedding2_normalized = embedding2 / (norm2 + 1e-8)

        # Compute cosine similarity
        similarity = torch.mean(
            torch.sum(embedding1_normalized * embedding2_normalized, dim=0)
        )

        return similarity.item()


def batch_create_node_representations(
    graphs: List[TemporalData], embd: Tensor, device: torch.device, node_dim: int
) -> Tensor:
    batch_size = len(graphs)
    
    all_msgs = torch.stack([graph.msg.to(device) for graph in graphs])
    all_times = torch.stack([graph.t.to(device) for graph in graphs])
    all_src = torch.stack([graph.src.to(device) for graph in graphs])
    all_dst = torch.stack([graph.dst.to(device) for graph in graphs])
    
    msg_min = all_msgs.min(dim=1, keepdim=True)[0]
    msg_max = all_msgs.max(dim=1, keepdim=True)[0]
    normalized_msgs = (all_msgs - msg_min) / (msg_max - msg_min + 1e-6)
    
    time_min = all_times.min(dim=1, keepdim=True)[0]
    time_max = all_times.max(dim=1, keepdim=True)[0]
    normalized_times = (all_times - time_min) / (time_max - time_min + 1e-6)
    
    num_events = normalized_msgs.shape[1]
    msg_dim = normalized_msgs.shape[2]
    
    batch_features = torch.zeros(batch_size, num_events * 2, msg_dim + 2, device=device)
    
    batch_features[:, :num_events, :msg_dim] = normalized_msgs
    batch_features[:, :num_events, msg_dim] = normalized_times
    batch_features[:, :num_events, msg_dim + 1] = 0
    
    batch_features[:, num_events:, :msg_dim] = normalized_msgs
    batch_features[:, num_events:, msg_dim] = normalized_times
    batch_features[:, num_events:, msg_dim + 1] = 1
    
    feature_dim = msg_dim + 2
    batch_projected = batch_features @ embd[:feature_dim, :]
    
    return batch_projected, all_src, all_dst

def create_node_representations(
    events: TemporalData, embd: Tensor
) -> Dict[int, Tensor]:
    node_reps = {}

    msg_min = events.msg.min(dim=0)[0]
    msg_max = events.msg.max(dim=0)[0]
    normalized_msgs = (events.msg - msg_min) / (
        msg_max - msg_min + 1e-6
    )

    time_min = events.t.min()
    time_max = events.t.max()
    normalized_times = (events.t - time_min) / (time_max - time_min + 1e-6)

    for i in range(events.src.size(0)):
        src, dst = int(events.src[i].item()), int(events.dst[i].item())
        msg = normalized_msgs[i]
        time_enc = normalized_times[i].reshape(1)

        combined_src = torch.cat([msg, time_enc, torch.tensor([0])], dim=0)
        combined_dst = torch.cat([msg, time_enc, torch.tensor([1])], dim=0)

        if src not in node_reps:
            node_reps[src] = []
        if dst not in node_reps:
            node_reps[dst] = []

        node_reps[src].append(combined_src)
        node_reps[dst].append(combined_dst)

    for node, reps in node_reps.items():
        node_reps[node] = torch.cat(reps) @ embd[: (reps[0].shape[0]) * len(reps)]

    return node_reps


def batch_jl_project_graph_level(batch_projected: Tensor, all_src: Tensor, all_dst: Tensor, proj_dim: int, device: torch.device) -> Tensor:
    batch_size, num_events_x2, node_dim = batch_projected.shape
    num_events = num_events_x2 // 2
    
    all_nodes = torch.cat([all_src[0], all_dst[0]]).unique().sort()[0]
    num_nodes = len(all_nodes)
    
    batch_node_matrices = torch.zeros(batch_size, num_nodes, node_dim, device=device)
    
    src_indices = all_src.long()
    dst_indices = all_dst.long()
    
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_events)
    
    batch_node_matrices.scatter_add_(1, 
        src_indices.unsqueeze(-1).expand(-1, -1, node_dim),
        batch_projected[:, :num_events, :])
    
    batch_node_matrices.scatter_add_(1,
        dst_indices.unsqueeze(-1).expand(-1, -1, node_dim), 
        batch_projected[:, num_events:, :])
    
    orthonormal_matrix = torch.randn((num_nodes, proj_dim), device=device) * torch.sqrt(
        torch.tensor(1.0 / num_nodes, device=device)
    )
    
    batch_graph_reps = torch.bmm(
        batch_node_matrices.transpose(1, 2),
        orthonormal_matrix.unsqueeze(0).expand(batch_size, -1, -1)
    )
    
    return batch_graph_reps

def jl_project_graph_level(node_reps: Dict[int, Tensor], proj_dim: int) -> Tensor:
    node_dim = list(node_reps.values())[0].size(0)
    num_nodes = len(node_reps.keys())

    node_reps = {index + 1: feature for index, feature in enumerate(node_reps.values())}

    orthonormal_matrix = torch.randn((num_nodes, proj_dim)) * torch.sqrt(
        torch.tensor(1.0 / num_nodes)
    )

    node_matrix = torch.zeros((num_nodes, node_dim))
    for node_id, rep in node_reps.items():
        node_matrix[int(node_id) - 1] = rep

    graph_rep = node_matrix.T @ orthonormal_matrix

    return graph_rep


def batch_jl_metric(
    graphs: List[TemporalData], 
    node_proj_dim: int, 
    graph_proj_dim: int, 
    seed: int,
    max_events: int = 100000,
    device: torch.device = torch.device('cpu')
) -> Tensor:
    torch_geometric.seed_everything(seed)
    
    batch_size = len(graphs)
    embd = torch.randn((max_events, node_proj_dim), device=device) * torch.sqrt(
        torch.tensor(1.0 / max_events, device=device)
    )
    
    batch_projected, all_src, all_dst = batch_create_node_representations(graphs, embd, device, node_proj_dim)
    batch_graph_reps = batch_jl_project_graph_level(batch_projected, all_src, all_dst, graph_proj_dim, device)
    
    return batch_graph_reps

def jl_metric(
    events: TemporalData, 
    node_proj_dim: int, 
    graph_proj_dim: int, 
    seed: int,
    max_events: int = 100000
) -> Tensor:
    torch_geometric.seed_everything(seed)

    embd = torch.randn((max_events, node_proj_dim)) * torch.sqrt(
        torch.tensor(1.0 / max_events)
    )

    node_reps = create_node_representations(events, embd)
    graph_rep = jl_project_graph_level(node_reps, graph_proj_dim)

    return graph_rep
