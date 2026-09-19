-- wrk2 Lua script: fixed system prompt (maximizes RadixAttention prefix cache hits)
-- All requests share the SAME system prompt → same prefix_hash → same node
-- → same SGLang instance → near 100% KV-Cache hit rate after warmup.
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

local FIXED_SYSTEM_PROMPT = "You are an expert AI assistant specializing in distributed systems and GPU computing. " ..
    "You provide concise, technically accurate answers. " ..
    "This is a long system prompt designed to fill the KV cache so we can measure the speedup from RadixAttention prefix caching."

local user_queries = {
    "What is RadixAttention?",
    "Explain Kafka consumer groups.",
    "How does prefix caching work in LLMs?",
    "What is the difference between TTFT and throughput?",
    "Describe the role of KV cache in transformer inference.",
}

local counter = 0

request = function()
    counter = counter + 1
    local query = user_queries[(counter % #user_queries) + 1]
    local body = string.format(
        '{"messages": [{"role": "system", "content": "%s"}, {"role": "user", "content": "%s"}]}',
        FIXED_SYSTEM_PROMPT, query)
    return wrk.format(nil, nil, nil, body)
end
