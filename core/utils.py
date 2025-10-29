import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from typing import List, Dict, Tuple


def mmr(doc_emb, cand_embs, cand_scores, k=3, lambda_param=0.7):
    if len(cand_embs) == 0:
        return []
    
    cand_embs = np.array(cand_embs)
    cand_scores = np.array(cand_scores)
    
    selected_ids = []
    cand_ids = list(range(len(cand_embs)))
    
    first = int(np.argmax(cand_scores))
    selected_ids.append(first)
    
    while len(selected_ids) < k and len(selected_ids) < len(cand_ids):
        remaining = [i for i in cand_ids if i not in selected_ids]
        if not remaining:
            break
            
        best = None
        best_score = -1e9
        
        for i in remaining:
            rel = cand_scores[i]
            
            if selected_ids:
                sims = [
                    cosine_similarity(
                        cand_embs[i:i+1], 
                        cand_embs[j:j+1]
                    )[0, 0]
                    for j in selected_ids
                ]
                sim_to_selected = max(sims)
            else:
                sim_to_selected = 0
            
            score = lambda_param * rel - (1 - lambda_param) * sim_to_selected
            
            if score > best_score:
                best_score = score
                best = i
        
        if best is not None:
            selected_ids.append(best)
        else:
            break
    
    return selected_ids


def normalize_emotion_vector(emotion_vec, min_threshold=0.01):
    emotion_vec = np.array(emotion_vec)
    emotion_vec[emotion_vec < min_threshold] = 0
    
    norm = np.linalg.norm(emotion_vec)
    if norm > 0:
        emotion_vec = emotion_vec / norm
    
    return emotion_vec


def top_k_emotions(emotion_vec, emotion_labels, k=5):
    emotion_vec = np.array(emotion_vec)
    top_indices = np.argsort(emotion_vec)[-k:][::-1]
    
    return [
        (emotion_labels[i], float(emotion_vec[i]))
        for i in top_indices
        if emotion_vec[i] > 0.01
    ]


def beam_search_decode(query_embedding, emotion_vector, emoji_embeddings, emoji_data, 
                       max_length=5, beam_width=5):
    """Beam search for optimal emoji sequence generation"""
    beams = [([], 0.0, set())]
    
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    emoji_norm = emoji_embeddings / (np.linalg.norm(emoji_embeddings, axis=1, keepdims=True) + 1e-8)
    base_scores = np.dot(emoji_norm, query_norm)
    
    for step in range(max_length):
        candidates = []
        
        for sequence, score, used_indices in beams:
            available_indices = [i for i in range(len(emoji_data)) if i not in used_indices]
            
            if len(available_indices) == 0:
                candidates.append((sequence, score, used_indices))
                continue
            
            candidate_scores = base_scores[available_indices].copy()
            
            if len(sequence) > 0:
                last_emoji_emb = emoji_embeddings[sequence[-1]]
                similarities = np.dot(
                    emoji_norm[available_indices],
                    last_emoji_emb / (np.linalg.norm(last_emoji_emb) + 1e-8)
                )
                diversity_bonus = 1 - similarities
                candidate_scores = candidate_scores * (0.7 + 0.3 * diversity_bonus)
            
            top_k = min(beam_width, len(available_indices))
            top_indices = np.argsort(candidate_scores)[::-1][:top_k]
            
            for idx in top_indices:
                emoji_idx = available_indices[idx]
                new_sequence = sequence + [emoji_idx]
                new_score = score + candidate_scores[idx]
                new_used = used_indices | {emoji_idx}
                candidates.append((new_sequence, new_score, new_used))
        
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return [(seq, score) for seq, score, _ in beams]


def submodular_mmr(query_embedding, emoji_embeddings, emoji_indices, k=5, lambda_param=0.5):
    """Enhanced MMR using submodular optimization"""
    if len(emoji_indices) <= k:
        return emoji_indices
    
    selected = []
    remaining = list(emoji_indices)
    
    candidate_embeddings = emoji_embeddings[emoji_indices]
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    relevance_scores = np.dot(
        candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8),
        query_norm
    )
    
    for _ in range(min(k, len(emoji_indices))):
        best_score = float('-inf')
        best_idx = None
        
        for i, candidate_idx in enumerate(remaining):
            relevance = relevance_scores[i]
            
            if len(selected) > 0:
                selected_embeddings = emoji_embeddings[selected]
                candidate_emb = emoji_embeddings[candidate_idx]
                similarities = np.dot(
                    selected_embeddings / (np.linalg.norm(selected_embeddings, axis=1, keepdims=True) + 1e-8),
                    candidate_emb / (np.linalg.norm(candidate_emb) + 1e-8)
                )
                redundancy = np.max(similarities)
            else:
                redundancy = 0
            
            score = relevance - lambda_param * redundancy
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx is not None:
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
            relevance_scores = np.delete(relevance_scores, best_idx)
    
    return np.array(selected)


def compute_semantic_coherence(emoji_sequence, query_embedding, emoji_embeddings, emoji_indices):
    """Measure emoji sequence meaningfulness"""
    if len(emoji_sequence) < 2:
        return {'overall': 1.0, 'pairwise': 1.0, 'global': 1.0}
    
    coherence_scores = []
    for i in range(len(emoji_indices) - 1):
        emb1 = emoji_embeddings[emoji_indices[i]]
        emb2 = emoji_embeddings[emoji_indices[i + 1]]
        similarity = np.dot(
            emb1 / (np.linalg.norm(emb1) + 1e-8),
            emb2 / (np.linalg.norm(emb2) + 1e-8)
        )
        coherence_scores.append(similarity)
    
    pairwise_coherence = np.mean(coherence_scores)
    
    sequence_embedding = np.mean([emoji_embeddings[idx] for idx in emoji_indices], axis=0)
    global_coherence = np.dot(
        sequence_embedding / (np.linalg.norm(sequence_embedding) + 1e-8),
        query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    )
    
    overall = 0.6 * pairwise_coherence + 0.4 * global_coherence
    
    return {
        'overall': overall,
        'pairwise': pairwise_coherence,
        'global': global_coherence
    }


def optimize_diversity_relevance(query_embedding, emoji_embeddings, candidate_indices, k=5):
    """Find optimal lambda for diversity-relevance trade-off"""
    lambda_values = np.linspace(0, 1, 10)
    best_lambda = 0.5
    best_f1 = 0
    
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    emoji_norm = emoji_embeddings / (np.linalg.norm(emoji_embeddings, axis=1, keepdims=True) + 1e-8)
    
    for lambda_param in lambda_values:
        selected = mmr(query_embedding, emoji_embeddings, 
                      np.dot(emoji_norm, query_norm), k=k, lambda_param=lambda_param)
        
        selected_embs = emoji_norm[selected]
        relevance = np.mean(np.dot(selected_embs, query_norm))
        
        if len(selected) > 1:
            similarity_matrix = np.dot(selected_embs, selected_embs.T)
            n = len(selected)
            diversity = 1 - (similarity_matrix.sum() - n) / (n * (n - 1) + 1e-8)
        else:
            diversity = 1.0
        
        f1 = 2 * (relevance * diversity) / (relevance + diversity + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_lambda = lambda_param
    
    return best_lambda


def emotion_vector_interpolation(emotion_vec1, emotion_vec2, alpha=0.5, method='spherical'):
    """Smooth interpolation between emotion states"""
    if method == 'linear':
        return (1 - alpha) * emotion_vec1 + alpha * emotion_vec2
    
    v1_norm = emotion_vec1 / (np.linalg.norm(emotion_vec1) + 1e-8)
    v2_norm = emotion_vec2 / (np.linalg.norm(emotion_vec2) + 1e-8)
    
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    theta = np.arccos(dot)
    
    if abs(theta) < 1e-6:
        return (1 - alpha) * emotion_vec1 + alpha * emotion_vec2
    
    sin_theta = np.sin(theta)
    interp = (
        np.sin((1 - alpha) * theta) / sin_theta * emotion_vec1 +
        np.sin(alpha * theta) / sin_theta * emotion_vec2
    )
    
    return interp


def calibrate_temperature(logits, true_labels, n_bins=10):
    """Temperature scaling for confidence calibration"""
    temperatures = np.linspace(0.5, 3.0, 50)
    best_ece = float('inf')
    best_temp = 1.0
    
    for temp in temperatures:
        scaled_logits = logits / temp
        probs = 1 / (1 + np.exp(-scaled_logits))
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if not np.any(in_bin):
                continue
            
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(true_labels[in_bin])
            bin_weight = np.sum(in_bin) / len(probs)
            ece += bin_weight * abs(avg_confidence - avg_accuracy)
        
        if ece < best_ece:
            best_ece = ece
            best_temp = temp
    
    return best_temp, best_ece


def batch_cosine_similarity(query_vectors, target_vectors):
    """Vectorized cosine similarity using broadcasting"""
    query_norm = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target_vectors, axis=1, keepdims=True)
    
    normalized_queries = query_vectors / (query_norm + 1e-8)
    normalized_targets = target_vectors / (target_norm + 1e-8)
    
    return np.dot(normalized_queries, normalized_targets.T)


def tsne_2d_projection(embeddings, perplexity=30, n_iter=1000):
    """Reduce embeddings to 2D using t-SNE for visualization"""
    try:
        from sklearn.manifold import TSNE
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42
        )
        
        return tsne.fit_transform(embeddings)
    
    except ImportError:
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        return centered @ eigenvectors[:, :2].real


def cluster_embeddings(embeddings_2d, n_clusters=10):
    """Identify emoji clusters in 2D space"""
    try:
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings_2d)
        return labels
    
    except ImportError:
        min_x, max_x = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        min_y, max_y = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        
        grid_size = int(np.sqrt(n_clusters))
        
        labels = []
        for pos in embeddings_2d:
            x_bin = int((pos[0] - min_x) / (max_x - min_x + 1e-8) * grid_size)
            y_bin = int((pos[1] - min_y) / (max_y - min_y + 1e-8) * grid_size)
            cluster_id = min(y_bin * grid_size + x_bin, n_clusters - 1)
            labels.append(cluster_id)
        
        return np.array(labels)


def compute_word_importance_gradients(model, tokenizer, text, emotion_idx):
    """Gradient-based word importance for emotion prediction"""
    import torch
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    embeddings = model.encoder.embeddings(input_ids) if hasattr(model.encoder, 'embeddings') else model.encoder.get_input_embeddings()(input_ids)
    embeddings.requires_grad = True
    
    outputs = model.encoder(inputs_embeds=embeddings, attention_mask=attention_mask, return_dict=True)
    
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        pooled = outputs.pooler_output
    else:
        pooled = outputs.last_hidden_state[:, 0]
    
    logits = model.head(pooled)
    probs = torch.sigmoid(logits)
    
    target_score = probs[0, emotion_idx]
    target_score.backward()
    
    gradients = embeddings.grad[0].abs().mean(dim=1).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    word_importance = [
        {'token': token, 'importance': float(gradients[i]), 'position': i}
        for i, token in enumerate(tokens)
        if token not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    
    word_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return word_importance


def generate_emotion_path(start_emotion, end_emotion, n_steps=10, method='spherical'):
    """Generate smooth emotion transition path"""
    alphas = np.linspace(0, 1, n_steps)
    
    path = []
    for alpha in alphas:
        interp = emotion_vector_interpolation(start_emotion, end_emotion, alpha, method)
        path.append(interp)
    
    return path


def analyze_emotion_trajectory(emotion_sequence, emotion_labels):
    """Analyze trajectory through emotion space"""
    if len(emotion_sequence) < 2:
        return None
    
    distances = []
    for i in range(len(emotion_sequence) - 1):
        dist = np.linalg.norm(emotion_sequence[i + 1] - emotion_sequence[i])
        distances.append(dist)
    
    total_distance = sum(distances)
    
    dominant_emotions = []
    for vec in emotion_sequence:
        top_idx = np.argmax(vec)
        dominant_emotions.append(emotion_labels[top_idx])
    
    return {
        'total_distance': float(total_distance),
        'average_step': float(np.mean(distances)),
        'max_step': float(np.max(distances)),
        'dominant_emotions': dominant_emotions,
        'trajectory_length': len(emotion_sequence)
    }
