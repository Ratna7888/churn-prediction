"""
Send realistic prediction requests to the API using rows from the actual dataset.
Generates traffic so Prometheus/Grafana dashboards have data to display.

Usage:
    python -m scripts.load_test                 # 50 requests, 0.2s apart
    python -m scripts.load_test --n 200 --delay 0.1
"""

import argparse
import time
import random
import requests
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("load_test")

API_URL = "http://localhost:8000/predict"

# These are the fields the API expects (everything except customerID and Churn)
REQUEST_FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def main():
    parser = argparse.ArgumentParser(description="Load test the churn API")
    parser.add_argument("--n", type=int, default=50, help="Number of requests")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv", help="Dataset path")
    args = parser.parse_args()

    # Load dataset and sample random rows
    df = pd.read_csv(args.data)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Convert SeniorCitizen to int in case it loaded as float
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    sample = df.sample(n=min(args.n, len(df)), random_state=None).reset_index(drop=True)

    logger.info(f"Sending {len(sample)} requests to {API_URL}")
    logger.info(f"Delay between requests: {args.delay}s")

    results = {"churn": 0, "retained": 0, "errors": 0}
    latencies = []

    for i, row in sample.iterrows():
        payload = {field: row[field] for field in REQUEST_FIELDS}

        # Ensure correct types for JSON
        payload["SeniorCitizen"] = int(payload["SeniorCitizen"])
        payload["tenure"] = int(payload["tenure"])
        payload["MonthlyCharges"] = float(payload["MonthlyCharges"])
        payload["TotalCharges"] = float(payload["TotalCharges"])

        try:
            start = time.perf_counter()
            resp = requests.post(API_URL, json=payload, timeout=5)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            if resp.status_code == 200:
                data = resp.json()
                label = "churn" if data["will_churn"] else "retained"
                results[label] += 1

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"[{i+1}/{len(sample)}] "
                        f"prob={data['churn_probability']:.3f} "
                        f"risk={data['risk_level']} "
                        f"latency={elapsed:.3f}s"
                    )
            else:
                results["errors"] += 1
                logger.warning(f"[{i+1}] HTTP {resp.status_code}: {resp.text[:100]}")

        except requests.exceptions.ConnectionError:
            results["errors"] += 1
            logger.error(f"[{i+1}] Connection failed — is the API running?")
            break
        except Exception as e:
            results["errors"] += 1
            logger.error(f"[{i+1}] Error: {e}")

        time.sleep(args.delay)

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Load test complete")
    logger.info(f"{'='*50}")
    logger.info(f"Total requests: {sum(results.values())}")
    logger.info(f"Churn predictions: {results['churn']}")
    logger.info(f"Retained predictions: {results['retained']}")
    logger.info(f"Errors: {results['errors']}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        logger.info(f"Avg latency: {avg:.3f}s | P95: {p95:.3f}s")


if __name__ == "__main__":
    main()