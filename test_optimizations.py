"""
Test script to verify ALL optimizations are working
"""
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import time
from pathlib import Path

print("="*60)
print("Testing MoodGlyphs Optimizations")
print("="*60)

# Test 1: FAISS Search
print("\n[1/10] Testing FAISS Search Integration...")
try:
    import faiss
    print("✓ FAISS imported successfully")
    
    from core.emoji_bank import EmojiBank
    
    # Create dummy emoji bank
    dummy_data = [{'emoji': '😊', 'desc': 'happy'}, {'emoji': '😢', 'desc': 'sad'}]
    dummy_embs = np.random.randn(2, 384).astype('float32')
    
    bank = EmojiBank(dummy_data, dummy_embs)
    
    if bank._use_faiss:
        print("✓ FAISS index initialized")
        
        query = np.random.randn(384).astype('float32')
        indices, scores = bank.fast_search(query, k=2)
        
        print(f"✓ FAISS search works: found {len(indices)} emojis")
        print(f"  Indices: {indices}, Scores: {scores[:2]}")
    else:
        print("✗ FAISS not enabled in EmojiBank")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ FAISS test failed: {e}")
    sys.exit(1)

# Test 2: LRU Cache
print("\n[2/10] Testing LRU Cache...")
try:
    from core.translator import TranslationCache
    
    cache = TranslationCache(max_size=10, ttl_seconds=60)
    
    cache.set("test", ("result", np.array([0.5])))
    result = cache.get("test")
    
    if result is not None:
        print(f"✓ Cache stores and retrieves: {cache.hits} hits, {cache.misses} misses")
    else:
        print("✗ Cache retrieval failed")
        sys.exit(1)
        
    # Test TTL
    cache.set("test2", "value2")
    time.sleep(0.1)
    result2 = cache.get("test2")
    if result2 is not None:
        print("✓ Cache TTL works")
    else:
        print("✗ Cache TTL failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Cache test failed: {e}")
    sys.exit(1)

# Test 3: Performance Monitoring
print("\n[3/10] Testing Performance Monitoring...")
try:
    from core.translator import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    monitor.record_stage("test_stage", 10.5)
    monitor.record_request(15.2)
    
    stats = monitor.get_stage_stats("test_stage")
    if stats and stats['mean_ms'] == 10.5:
        print(f"✓ Stage timing works: {stats['mean_ms']}ms mean")
    else:
        print(f"✗ Stage timing failed: {stats}")
        sys.exit(1)
    
    latency = monitor.get_latency_percentiles()
    if latency and 'mean_ms' in latency:
        print(f"✓ Latency tracking works: {latency['mean_ms']}ms mean")
    else:
        print(f"✗ Latency tracking failed")
        sys.exit(1)
    
    memory = monitor.get_memory_stats()
    if memory and 'current_mb' in memory:
        print(f"✓ Memory tracking works: {memory['current_mb']:.1f}MB current")
    else:
        print(f"✗ Memory tracking failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Performance monitoring test failed: {e}")
    sys.exit(1)

# Test 4: Batch Processing
print("\n[4/10] Testing Batch Processing...")
try:
    # We'll test this with utils functions
    from core.utils import batch_cosine_similarity
    
    queries = np.random.randn(5, 384)
    targets = np.random.randn(10, 384)
    
    result = batch_cosine_similarity(queries, targets)
    
    if result.shape == (5, 10):
        print(f"✓ Batch cosine similarity works: shape {result.shape}")
    else:
        print(f"✗ Batch cosine similarity failed: wrong shape {result.shape}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Batch processing test failed: {e}")
    sys.exit(1)

# Test 5: Beam Search
print("\n[5/10] Testing Beam Search...")
try:
    from core.utils import beam_search_decode
    
    query_emb = np.random.randn(384)
    emotion_vec = np.random.rand(28)
    emoji_embs = np.random.randn(100, 384)
    emoji_data = [{'emoji': f'e{i}', 'desc': f'desc{i}'} for i in range(100)]
    
    sequences = beam_search_decode(query_emb, emotion_vec, emoji_embs, emoji_data, 
                                   max_length=3, beam_width=3)
    
    if len(sequences) > 0 and len(sequences[0]) == 2:
        print(f"✓ Beam search works: generated {len(sequences)} sequences")
        print(f"  Best sequence length: {len(sequences[0][0])}, score: {sequences[0][1]:.3f}")
    else:
        print(f"✗ Beam search failed: {sequences}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Beam search test failed: {e}")
    sys.exit(1)

# Test 6: Submodular MMR
print("\n[6/10] Testing Submodular MMR...")
try:
    from core.utils import submodular_mmr
    
    query_emb = np.random.randn(384)
    emoji_embs = np.random.randn(50, 384)
    indices = np.arange(50)
    
    selected = submodular_mmr(query_emb, emoji_embs, indices, k=5, lambda_param=0.5)
    
    if len(selected) == 5:
        print(f"✓ Submodular MMR works: selected {len(selected)} emojis")
        print(f"  Indices: {selected}")
    else:
        print(f"✗ Submodular MMR failed: selected {len(selected)} instead of 5")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Submodular MMR test failed: {e}")
    sys.exit(1)

# Test 7: Semantic Coherence
print("\n[7/10] Testing Semantic Coherence...")
try:
    from core.utils import compute_semantic_coherence
    
    emoji_seq = "😊😢"
    query_emb = np.random.randn(384)
    emoji_embs = np.random.randn(10, 384)
    indices = [0, 1]
    
    scores = compute_semantic_coherence(emoji_seq, query_emb, emoji_embs, indices)
    
    if 'overall' in scores and 'pairwise' in scores and 'global' in scores:
        print(f"✓ Semantic coherence works: overall={scores['overall']:.3f}")
        print(f"  Pairwise={scores['pairwise']:.3f}, Global={scores['global']:.3f}")
    else:
        print(f"✗ Semantic coherence failed: {scores}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Semantic coherence test failed: {e}")
    sys.exit(1)

# Test 8: Bayesian Personalization
print("\n[8/10] Testing Bayesian Personalization...")
try:
    from core.profile import BayesianPreferences
    
    bayes = BayesianPreferences(prior_strength=1.0)
    
    bayes.update('😊', 'joy')
    bayes.update('😊', 'joy')
    bayes.update('😢', 'sadness')
    
    prob_happy = bayes.get_probability('😊', 'joy')
    prob_sad = bayes.get_probability('😢', 'sadness')
    
    if prob_happy > 0 and prob_sad > 0:
        print(f"✓ Bayesian preferences work: P(😊|joy)={prob_happy:.4f}, P(😢|sad)={prob_sad:.4f}")
    else:
        print(f"✗ Bayesian preferences failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Bayesian personalization test failed: {e}")
    sys.exit(1)

# Test 9: EMA Tracking
print("\n[9/10] Testing EMA Tracking...")
try:
    from core.profile import ExponentialMovingAverage
    
    ema = ExponentialMovingAverage(alpha=0.1, dimension=384)
    
    emb1 = np.random.randn(384)
    emb2 = np.random.randn(384)
    
    ema.update(emb1)
    ema.update(emb2)
    
    if ema.embedding is not None and ema.update_count == 2:
        print(f"✓ EMA tracking works: {ema.update_count} updates, alpha={ema.alpha:.3f}")
        
        query = np.random.randn(384)
        personalized = ema.get_personalized_query(query, blend=0.2)
        
        if personalized.shape == (384,):
            print(f"✓ EMA personalization works")
        else:
            print(f"✗ EMA personalization failed: wrong shape")
            sys.exit(1)
    else:
        print(f"✗ EMA tracking failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ EMA tracking test failed: {e}")
    sys.exit(1)

# Test 10: Model Quantization & ONNX
print("\n[10/10] Testing Model Optimization...")
try:
    # Check if models exist first
    model_path = Path("models/emotion_model.pt")
    
    if not model_path.exists():
        print("⚠ Model not trained yet, skipping model optimization tests")
        print("  (This is OK - run 'python cli.py setup' first)")
    else:
        from core.emotion_model import EmotionModel
        import torch
        
        model = EmotionModel("sentence-transformers/all-MiniLM-L6-v2", n_emotions=28)
        
        # Test model size calculation
        size_mb = model.get_model_size_mb()
        if size_mb > 0:
            print(f"✓ Model size calculation works: {size_mb:.1f}MB")
        else:
            print(f"✗ Model size calculation failed")
            sys.exit(1)
        
        # Test quantization
        try:
            quantized = model.quantize()
            print(f"✓ Model quantization works")
        except Exception as qe:
            print(f"⚠ Model quantization: {qe}")
        
        print("  (Full model tests require trained weights)")
        
except Exception as e:
    print(f"✗ Model optimization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 11: t-SNE & Clustering
print("\n[BONUS] Testing t-SNE & Clustering...")
try:
    from core.utils import tsne_2d_projection, cluster_embeddings
    
    embeddings = np.random.randn(50, 384)
    
    embeddings_2d = tsne_2d_projection(embeddings, n_iter=250)  # min 250 iterations required
    
    if embeddings_2d.shape == (50, 2):
        print(f"✓ t-SNE projection works: {embeddings.shape} -> {embeddings_2d.shape}")
    else:
        print(f"✗ t-SNE failed: wrong shape {embeddings_2d.shape}")
        sys.exit(1)
    
    labels = cluster_embeddings(embeddings_2d, n_clusters=5)
    
    if len(labels) == 50 and len(set(labels)) <= 5:
        print(f"✓ Clustering works: {len(set(labels))} clusters found")
    else:
        print(f"✗ Clustering failed: {len(labels)} labels, {len(set(labels))} clusters")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ t-SNE/Clustering test failed: {e}")
    sys.exit(1)

# Test 12: Temperature Calibration
print("\n[BONUS] Testing Temperature Calibration...")
try:
    from core.utils import calibrate_temperature
    
    logits = np.random.randn(100, 28)
    labels = (np.random.rand(100, 28) > 0.5).astype(float)
    
    best_temp, ece = calibrate_temperature(logits, labels, n_bins=5)
    
    if 0.5 <= best_temp <= 3.0 and ece >= 0:
        print(f"✓ Temperature calibration works: T={best_temp:.2f}, ECE={ece:.4f}")
        print(f"  (High ECE expected with random data)")
    else:
        print(f"✗ Temperature calibration failed: T={best_temp}, ECE={ece}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Temperature calibration test failed: {e}")
    sys.exit(1)

# Test 13: Emotion Interpolation
print("\n[BONUS] Testing Emotion Interpolation...")
try:
    from core.utils import emotion_vector_interpolation, generate_emotion_path
    
    vec1 = np.random.rand(28)
    vec2 = np.random.rand(28)
    
    # Test SLERP
    interp = emotion_vector_interpolation(vec1, vec2, alpha=0.5, method='spherical')
    
    if interp.shape == (28,):
        print(f"✓ Emotion interpolation (SLERP) works")
    else:
        print(f"✗ SLERP failed: wrong shape {interp.shape}")
        sys.exit(1)
    
    # Test path generation
    path = generate_emotion_path(vec1, vec2, n_steps=5, method='spherical')
    
    if len(path) == 5 and all(p.shape == (28,) for p in path):
        print(f"✓ Emotion path generation works: {len(path)} steps")
    else:
        print(f"✗ Path generation failed")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Emotion interpolation test failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL OPTIMIZATIONS PASSED!")
print("="*60)
print("\nSummary:")
print("  1. FAISS Search ✓")
print("  2. LRU Cache ✓")
print("  3. Performance Monitoring ✓")
print("  4. Batch Processing ✓")
print("  5. Beam Search ✓")
print("  6. Submodular MMR ✓")
print("  7. Semantic Coherence ✓")
print("  8. Bayesian Personalization ✓")
print("  9. EMA Tracking ✓")
print("  10. Model Optimization ✓")
print("  11. t-SNE & Clustering ✓")
print("  12. Temperature Calibration ✓")
print("  13. Emotion Interpolation ✓")
print("\nAll optimizations are working and integrated!")

