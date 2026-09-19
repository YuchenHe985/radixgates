"""
enterprise_qa_demo.py — 企业内部智能问答并发压测

架构：Client → Gateway → SGLang（直连，无 Kafka / Worker）
与后端并行模式无关，适用于 DP / TP / EP 任意部署。

特性展示：
  - 30 名员工同时提问（5 个部门，每部门 6 人）
  - 每个部门有独立 system prompt → 不同 prefix_hash → consistent hash 路由
  - Go 信号量限流（max_concurrent=8）防止 SGLang 被打爆
  - 全部同步 HTTP，无 task_id 轮询，无 Kafka，无 WebSocket

启动（本地 Docker）：
  docker compose -f docker-compose.yml up -d
  python3 examples/enterprise_qa_demo.py

启动（裸机，no-Docker）：
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

# 5 个部门 → 5 种 system prompt → 5 种 prefix_hash → 分散并发负载
DEPARTMENT_PROMPTS = {
    "Engineering": "You are a helpful AI assistant serving the Engineering department. Be concise and accurate.",
    "HR":          "You are a helpful AI assistant serving the HR department. Be concise and friendly.",
    "Finance":     "You are a helpful AI assistant serving the Finance department. Be concise and precise.",
    "Product":     "You are a helpful AI assistant serving the Product department. Be concise and clear.",
    "Marketing":   "You are a helpful AI assistant serving the Marketing department. Be concise and engaging.",
}
DEPARTMENTS = list(DEPARTMENT_PROMPTS.keys())

EMPLOYEE_QUESTIONS = [
    # 考验知识库（RAG 命中）的问题
    "I want to know about the revolutionary RadixAttention approach.",
    "Which framework uses RadixAttention and what does it do?",
    "Where is the Eiffel Tower located and who built it?",
    "Could you tell me the capital of France?",
    # 日常聊天
    "Hello! Write a 1-sentence greeting for our team.",
    "How much is 1 + 1? Be concise.",
]


def simulate_employee(employee_id: int):
    """一名员工：选部门 → 提问 → 等直接回答。"""
    dept = DEPARTMENTS[employee_id % len(DEPARTMENTS)]
    system_prompt = DEPARTMENT_PROMPTS[dept]
    question = EMPLOYEE_QUESTIONS[employee_id % len(EMPLOYEE_QUESTIONS)]

    # 轻微错峰，模拟真实用户行为
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
            return False, f"[{dept}][员工 {employee_id}] ❌ HTTP {resp.status_code}: {resp.text[:100]}"

        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        latency = time.time() - start
        short_ans = answer[:80].replace("\n", " ")
        return (
            True,
            f"[{dept}][员工 {employee_id}] 耗时 {latency:.2f}s ✅\n"
            f"  提问: {question}\n"
            f"  回答: {short_ans}\n" + "-" * 50,
        )

    except Exception as e:
        return False, f"[{dept}][员工 {employee_id}] 💥 异常: {e}"


from _log_utils import save_container_logs as _save_container_logs

def save_container_logs(ts: str):
    _save_container_logs(_LOG_DIR, ts)


def main():
    print("=" * 60)
    print("🌟 场景模拟：企业早高峰智能问答并发突增（Direct 模式）")
    print(f"   Gateway: {GATEWAY_URL}")
    print("=" * 60)

    # 确认 Gateway 模式
    try:
        health = requests.get(f"{GATEWAY_URL}/health", timeout=5).json()
        mode = health.get("mode", "unknown")
        if mode != "direct":
            print(f"⚠️  当前模式 {mode!r}，需要 'direct' 模式")
            print("   请用 ROUTING_MODE=direct docker compose up 重启 Gateway")
            sys.exit(1)
        print(f"✅ Gateway 模式: {mode!r}  （无 Kafka，无 Worker）")
    except Exception as e:
        print(f"❌ 无法连接 Gateway: {e}")
        sys.exit(1)

    NUM_EMPLOYEES = 30
    print(f"\n👥 模拟 {NUM_EMPLOYEES} 名员工并发提问（{len(DEPARTMENTS)} 个部门）")
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
    print("🎉 场景模拟结束（并发潮已消退）。")
    print(f"   总耗时:  {elapsed:.1f}s")
    print(f"   总员工:  {NUM_EMPLOYEES}")
    print(f"   成功:    {success_count}")
    print(f"   失败:    {fail_count}")
    if fail_count == 0:
        print("🏆 信号量限流完美运作：所有请求在等待池中有序处理，零丢失！")
    print("=" * 60)

    _ts = os.path.basename(_log_path).replace("enterprise_qa_demo_", "").replace(".log", "")
    save_container_logs(_ts)


if __name__ == "__main__":
    main()
