import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
