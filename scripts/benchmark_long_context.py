import urllib.request
import json
import time
import os

def chat_stream_ttft(context_text, question, port=30000):
    messages = [
        {'role': 'system', 'content': f'Context: {context_text}'},
        {'role': 'user', 'content': question}
    ]
    payload = json.dumps({
        'model': 'NousResearch/Meta-Llama-3-8B-Instruct',
        'messages': messages,
        'stream': True,
        'max_tokens': 10
    }).encode()
    
    req = urllib.request.Request(f'http://localhost:{port}/v1/chat/completions', data=payload, method='POST')
    req.add_header('Content-Type', 'application/json')
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            first_chunk_time = None
            for line in response:
                if line.strip() and first_chunk_time is None:
                    first_chunk_time = time.time()
                    break # 我们只关心 TTFT
        ttft_ms = (first_chunk_time - t0) * 1000 if first_chunk_time else 0
        return ttft_ms
    except Exception as e:
        print(f"Error getting TTFT: {e}")
        return None

def chat_non_stream_for_tokens(context_text, question, port=30000):
    messages = [
        {'role': 'system', 'content': f'Context: {context_text}'},
        {'role': 'user', 'content': question}
    ]
    payload = json.dumps({
        'model': 'NousResearch/Meta-Llama-3-8B-Instruct',
        'messages': messages,
        'stream': False,
        'max_tokens': 1
    }).encode()
    req = urllib.request.Request(f'http://localhost:{port}/v1/chat/completions', data=payload, method='POST')
    req.add_header('Content-Type', 'application/json')
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
        usage = resp.get('usage', {})
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        total = usage.get('prompt_tokens', 0)
        return cached, total
    except Exception as e:
        print(f"Error getting token counts: {e}")
        return 0, 0

print("==============================================================")
print("  RadixGates 🚀 Long-Context TTFT Benchmark (Cold vs Warm)")
print("==============================================================")

base_text = "The RadixAttention mechanism developed by SGLang dramatically optimizes inference in LLMs by utilizing a radix tree to manage the Key-Value (KV) cache. Instead of recalculating the attention scores for identical prefixes across multiple requests, the system intelligently hashes and retrieves previously computed states. This approach is highly effective in Retrieval-Augmented Generation (RAG) paradigms, where the same large document or system prompt is frequently prepended to varying user queries. "

# 构造不同规模的测试上下文
context_map = {
    "2K": base_text * 40,   # roughly 2000+ tokens
    "4K": base_text * 80,   # roughly 4000+ tokens
    "8K": base_text * 150,  # roughly 8000+ tokens
}

question = "Please summarize the core mechanism."

results_summary = []

for label, ctx in context_map.items():
    print(f"\n[ Testing Context Size: {label} ]")
    # 为了保证冷启动是纯天然的（以防以前测试跑过同样的），我们在前面动态塞入当前标签和时间戳来破坏前缀缓存的碰撞
    ctx_unique = f"Unique_Run_ID: {label}_{time.time()} \n" + ctx
    
    print(f"  👉 Running Cold...")
    cold_ttft = chat_stream_ttft(ctx_unique, question)
    
    if cold_ttft is None:
        print("  ❌ Failed to get cold response, skipping.")
        continue

    print(f"  🔥 Running Warm (3 iterations)...")
    warm_ttft_list = []
    for _ in range(3):
        ttft = chat_stream_ttft(ctx_unique, question)
        if ttft: 
            warm_ttft_list.append(ttft)
        time.sleep(0.1)  # 稍微停顿一下
    
    avg_warm_ttft = sum(warm_ttft_list) / len(warm_ttft_list) if warm_ttft_list else 0
    
    # 最后发一个非流式请求去拿到 SGLang API 返回的确切输入 token 数和 cache 命中数
    cached_tokens, total_tokens = chat_non_stream_for_tokens(ctx_unique, question)
    
    speedup = cold_ttft / avg_warm_ttft if avg_warm_ttft > 0 else 0
    hit_rate = (cached_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    print(f"  📊 Metrics for ~{total_tokens} actual tokens:")
    print(f"     Cold TTFT : {cold_ttft:.2f} ms")
    print(f"     Warm TTFT : {avg_warm_ttft:.2f} ms (avg)")
    print(f"     Speedup   : {speedup:.2f}x 🚀")
    print(f"     Cache Hit : {hit_rate:.1f}% ({cached_tokens}/{total_tokens})")
    
    results_summary.append({
        "Size": label,
        "Tokens": total_tokens,
        "Cold_TTFT": f"{cold_ttft:.2f}",
        "Warm_TTFT": f"{avg_warm_ttft:.2f}",
        "Speedup": f"{speedup:.2f}x",
        "Hit_Rate": f"{hit_rate:.1f}%"
    })

print("\n========================= SUMMARY =========================")
print(f"{'Size':<6} | {'Tokens':<8} | {'Cold TTFT':<12} | {'Warm TTFT':<12} | {'Speedup':<8} | {'Hit Rate'}")
print("-" * 75)
for res in results_summary:
    print(f"{res['Size']:<6} | {res['Tokens']:<8} | {res['Cold_TTFT']+' ms':<12} | {res['Warm_TTFT']+' ms':<12} | {res['Speedup']:<8} | {res['Hit_Rate']}")
print("===========================================================")
