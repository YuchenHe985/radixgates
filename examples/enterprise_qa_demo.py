"""
enterprise_qa_demo.py — concurrent internal-assistant workload.

Architecture: Client → Gateway → SGLang (direct HTTP; no Kafka or worker tier).
The workload can run against DP, TP, or EP deployments.

Workload:
  - 30 concurrent users across five departments
  - one system prompt per department to exercise prefix affinity
  - gateway admission limits protect each SGLang worker
  - synchronous HTTP with no task polling, Kafka, or WebSocket layer

Local Docker:
  docker compose -f docker-compose.yml up -d
  python3 examples/enterprise_qa_demo.py

Bare metal:
  GATEWAY_URL=http://localhost:8081 python3 examples/enterprise_qa_demo.py
"""

import concurrent.futures
import json
import os
import random
import sys
import time
from datetime import datetime

import requests

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")

_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_path = os.path.join(_LOG_DIR, f"enterprise_qa_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, "w", buffering=1)

class _Tee:
    def __init__(self, *targets): self._targets = targets
    def write(self, s):
        for t in self._targets: t.write(s)
    def flush(self):
        for t in self._targets: t.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"[log → {_log_path}]")

# Five departments produce five reusable system-prompt prefixes.
DEPARTMENT_PROMPTS = {
    "Engineering": "You are a helpful AI assistant serving the Engineering department. Be concise and accurate.",
    "HR":          "You are a helpful AI assistant serving the HR department. Be concise and friendly.",
    "Finance":     "You are a helpful AI assistant serving the Finance department. Be concise and precise.",
    "Product":     "You are a helpful AI assistant serving the Product department. Be concise and clear.",
    "Marketing":   "You are a helpful AI assistant serving the Marketing department. Be concise and engaging.",
}
DEPARTMENTS = list(DEPARTMENT_PROMPTS.keys())

EMPLOYEE_QUESTIONS = [
    # Questions that can exercise a retrieval-backed deployment.
    "I want to know about the revolutionary RadixAttention approach.",
    "Which framework uses RadixAttention and what does it do?",
    "Where is the Eiffel Tower located and who built it?",
    "Could you tell me the capital of France?",
    # General prompts.
    "Hello! Write a 1-sentence greeting for our team.",
    "How much is 1 + 1? Be concise.",
]


def simulate_employee(employee_id: int):
    """Submit one user request using a department-specific system prompt."""
    dept = DEPARTMENTS[employee_id % len(DEPARTMENTS)]
    system_prompt = DEPARTMENT_PROMPTS[dept]
    question = EMPLOYEE_QUESTIONS[employee_id % len(EMPLOYEE_QUESTIONS)]

    # Add jitter so requests do not all begin in the same scheduler tick.
    time.sleep(random.uniform(0, 2.0))
    start = time.time()

    try:
        resp = requests.post(
            f"{GATEWAY_URL}/v1/chat",
            json={
                "model": "NousResearch/Meta-Llama-3-8B-Instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ],
                "stream": False,
            },
            timeout=300,
        )

        if resp.status_code != 200:
            return False, f"[{dept}][user {employee_id}] ❌ HTTP {resp.status_code}: {resp.text[:100]}"

        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        latency = time.time() - start
        short_ans = answer[:80].replace("\n", " ")
        return (
            True,
            f"[{dept}][user {employee_id}] latency {latency:.2f}s ✅\n"
            f"  prompt: {question}\n"
            f"  answer: {short_ans}\n" + "-" * 50,
        )

    except Exception as e:
        return False, f"[{dept}][user {employee_id}] 💥 error: {e}"


from _log_utils import save_container_logs as _save_container_logs

def save_container_logs(ts: str):
    _save_container_logs(_LOG_DIR, ts)


def main():
    print("=" * 60)
    print("🌟 Internal-assistant concurrent workload (direct mode)")
    print(f"   Gateway: {GATEWAY_URL}")
    print("=" * 60)

    # Verify gateway mode.
    try:
        health = requests.get(f"{GATEWAY_URL}/health", timeout=5).json()
        mode = health.get("mode", "unknown")
        if mode != "direct":
            print(f"⚠️  Current mode is {mode!r}; this demo requires 'direct'")
            print("   Restart with ROUTING_MODE=direct docker compose up")
            sys.exit(1)
        print(f"✅ Gateway mode: {mode!r} (no Kafka or worker tier)")
    except Exception as e:
        print(f"❌ Cannot reach gateway: {e}")
        sys.exit(1)

    NUM_EMPLOYEES = 30
    print(f"\n👥 Sending {NUM_EMPLOYEES} requests across {len(DEPARTMENTS)} departments")
    print("─" * 60)

    success_count = 0
    fail_count = 0
    t_total = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_EMPLOYEES) as ex:
        futures = [ex.submit(simulate_employee, i) for i in range(NUM_EMPLOYEES)]
        for future in concurrent.futures.as_completed(futures):
            ok, msg = future.result()
            print(msg)
            if ok:
                success_count += 1
            else:
                fail_count += 1

    elapsed = time.time() - t_total
    print("=" * 60)
    print("🎉 Workload complete.")
    print(f"   Wall time: {elapsed:.1f}s")
    print(f"   Requests:  {NUM_EMPLOYEES}")
    print(f"   Success:   {success_count}")
    print(f"   Failed:    {fail_count}")
    if fail_count == 0:
        print("✅ All submitted requests completed in this run.")
    print("=" * 60)

    _ts = os.path.basename(_log_path).replace("enterprise_qa_demo_", "").replace(".log", "")
    save_container_logs(_ts)


if __name__ == "__main__":
    main()
