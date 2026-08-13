# run_eval.py
# Runs every question in the golden test set through the self-healing RAG
# and measures behaviour: does it answer when it should, refuse when it
# should, and how often does it hallucinate (answer the unanswerable)?

import json
import time
from graph import app   # our compiled self-healing graph

# --- Load the golden test set ---
with open("eval/testset.json", "r", encoding="utf-8") as f:
    testset = json.load(f)

# --- Heuristic refusal detector ---
# A simple keyword check. Good enough for this eval; a production system
# would use a model-based judge. Worth noting as a known limitation.
REFUSAL_PHRASES = [
    "don't have enough", "do not have enough", "don't know", "do not know",
    "not mention", "no information", "not provide", "not contain",
    "cannot answer", "can't answer", "unable to", "does not mention",
]

def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)

# --- Run every question through the system ---
results = []
total_latency = 0.0
print(f"Running {len(testset)} questions through the system...\n")

for item in testset:
    q = item["question"]
    answerable = item["answerable"]

    start = time.time()
    output = app.invoke({"question": q, "original_question": q, "retries": 0})
    latency = time.time() - start
    total_latency += latency

    answer = output["answer"]
    refused = is_refusal(answer)

    # answerable   -> should ANSWER (not refuse)
    # unanswerable -> should REFUSE
    correct = (not refused) if answerable else refused

    results.append({
        "question": q, "answerable": answerable, "refused": refused,
        "correct": correct, "latency_sec": round(latency, 2), "answer": answer,
    })
    print(f"  {'OK' if correct else 'XX'}  ({latency:4.1f}s)  {q}")

# --- Compute metrics ---
total = len(results)
unanswerable = [r for r in results if not r["answerable"]]
correct_count = sum(1 for r in results if r["correct"])
accuracy = correct_count / total * 100

# Hallucination = an unanswerable question that got a real (non-refusal) answer.
hallucinations = [r for r in unanswerable if not r["refused"]]
hallucination_rate = len(hallucinations) / len(unanswerable) * 100 if unanswerable else 0
refusal_accuracy = (len(unanswerable) - len(hallucinations)) / len(unanswerable) * 100 if unanswerable else 0
avg_latency = total_latency / total

# --- Print the report ---
print("\n" + "=" * 45)
print("            EVALUATION REPORT")
print("=" * 45)
print(f"Total questions        : {total}")
print(f"Overall accuracy       : {accuracy:.1f}%  ({correct_count}/{total})")
print(f"Hallucination rate     : {hallucination_rate:.1f}%   (target: under 5%)")
print(f"Correct-refusal rate   : {refusal_accuracy:.1f}%")
print(f"Avg latency per query  : {avg_latency:.1f}s")
print("=" * 45)

# --- Save results for later (dashboard / CI comparison) ---
with open("eval/last_run.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": round(accuracy, 1),
        "hallucination_rate": round(hallucination_rate, 1),
        "refusal_accuracy": round(refusal_accuracy, 1),
        "avg_latency_sec": round(avg_latency, 2),
        "results": results,
    }, f, indent=2)
print("\nDetailed results saved to eval/last_run.json")