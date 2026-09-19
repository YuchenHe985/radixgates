-- wrk2 Lua script: random user prompt (no prefix cache benefit)
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

local prompts = {
    "Explain the difference between TCP and UDP.",
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "What is gradient descent in machine learning?",
    "Describe the process of photosynthesis.",
    "What are the SOLID principles in software engineering?",
    "Explain how a transformer model works.",
    "What is the difference between a mutex and a semaphore?",
}

local counter = 0

request = function()
    counter = counter + 1
    local prompt = prompts[(counter % #prompts) + 1]
    local body = string.format(
        '{"messages": [{"role": "user", "content": "%s"}]}', prompt)
    return wrk.format(nil, nil, nil, body)
end

response = function(status, headers, body)
    if status ~= 202 then
        io.write("[wrk] Unexpected status: " .. status .. "\n")
    end
end
