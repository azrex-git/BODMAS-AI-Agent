"""
Component 3: 1-Click Evaluation Script (`harness/eval.py`)
Executes an automated test suite of edge-case math strings,
validates mathematical assertions, and outputs a formatted terminal report.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

from .config import DEFAULT_MODEL, get_api_key
from .pipeline import solve_bodmas_pipeline


def load_test_cases(file_path: str = None) -> List[Dict[str, Any]]:
    if not file_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "tests", "test_cases.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(model_id: str = DEFAULT_MODEL, api_key: str = None) -> Dict[str, Any]:
    test_cases = load_test_cases()
    results = []
    total_latency = 0.0
    total_tokens = 0
    passed_count = 0

    print("\n" + "=" * 90)
    print(" ▮ BODMAS TUTOR AGENT: GEMMA 4 BENCHMARK EVALUATION HARNESS ▮")
    print(f" Model: {model_id} | API: Google Gemini API | Mode: Multi-Stage Pipeline")
    print("=" * 90)
    print(f"{'Test ID':<8} | {'Expression':<26} | {'Expected':<10} | {'Actual':<10} | {'Status':<8} | {'Latency':<8} | {'Tokens':<6}")
    print("-" * 90)

    for case in test_cases:
        test_id = case["id"]
        equation = case["equation"]
        expected = case["expected_answer"]
        expect_valid = case["expect_valid"]

        res = solve_bodmas_pipeline(equation, model_id=model_id, api_key=api_key)
        telemetry = res.get("telemetry", {})
        latency_ms = telemetry.get("total_latency_ms", 0.0)
        tokens = telemetry.get("total_tokens", 0)

        total_latency += latency_ms
        total_tokens += tokens

        actual_status = res.get("status")
        actual_ans = res.get("final_answer", "")

        is_passed = False
        if not expect_valid:
            # Expected to be caught by Stage 1 guardrail
            if actual_status == "invalid_input":
                is_passed = True
                actual_display = "INVALID"
            else:
                actual_display = actual_ans or "VALID"
        else:
            actual_display = actual_ans
            # Compare numerical value (strip spaces or floating zeros)
            try:
                if float(actual_ans) == float(expected):
                    is_passed = True
            except (ValueError, TypeError):
                if str(actual_ans).strip() == str(expected).strip():
                    is_passed = True

        status_str = "PASSED" if is_passed else "FAILED"
        if is_passed:
            passed_count += 1

        print(f"{test_id:<8} | {equation[:24]:<26} | {expected:<10} | {actual_display:<10} | {status_str:<8} | {latency_ms:>6.1f}ms | {tokens:>6}")

        results.append({
            "id": test_id,
            "name": case["name"],
            "equation": equation,
            "expected": expected,
            "actual": actual_display,
            "passed": is_passed,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "telemetry": telemetry,
        })

    avg_latency = round(total_latency / len(test_cases), 2) if test_cases else 0.0
    pass_percentage = round((passed_count / len(test_cases)) * 100, 1) if test_cases else 0.0

    print("-" * 90)
    print(f" SUMMARY: {passed_count}/{len(test_cases)} Passed ({pass_percentage}%) | Avg Latency: {avg_latency} ms | Total Tokens: {total_tokens}")
    print("=" * 90 + "\n")

    return {
        "model": model_id,
        "total_tests": len(test_cases),
        "passed_tests": passed_count,
        "pass_percentage": pass_percentage,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "results": results,
    }


if __name__ == "__main__":
    cli_key = sys.argv[1] if len(sys.argv) > 1 else None
    cli_model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    run_evaluation(model_id=cli_model, api_key=cli_key)
