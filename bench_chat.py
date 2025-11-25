# bench_chat.py
import requests, time, statistics, json

URL = "http://127.0.0.1:8000/chat"
# Use short representative queries to keep generation short
QUERIES = [
    "How do I reset my password?",
    "What is the refund policy?",
    "How do I change my account email?",
    "How to contact support?",
    "How to cancel my subscription?",
    "Where can I view my invoices?",
    "How to update billing info?",
    "What are your working hours?",
    "How to change my username?",
    "How to subscribe to premium?"
] * 5  # 50 total

times = []
failures = 0
detailed = []

for q in QUERIES:
    t0 = time.time()
    try:
        r = requests.post(URL, json={"text": q}, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        failures += 1
        print("Request error for query:", q, "error:", e)
        detailed.append({"query": q, "error": str(e)})
        continue
    t1 = time.time()
    ms = (t1 - t0) * 1000
    times.append(ms)
    detailed.append({"query": q, "time_ms": int(ms), "source": data.get("source"), "timing_ms": data.get("timing_ms")})
    # small sleep to avoid flooding
    time.sleep(0.1)

if not times:
    print("No successful requests recorded. Failures:", failures)
else:
    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(len(times) * 0.95) - 1]
    print("samples:", len(times), "failures:", failures, "P50(ms):", int(p50), "P95(ms):", int(p95))
    # show a few example response timings
    print("\nSample detailed results (first 8):")
    for item in detailed[:8]:
        print(json.dumps(item, ensure_ascii=False))
