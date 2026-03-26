"""Prime Code Generation + Execution Validator.

Sends coding tasks to Prime, extracts Python code from the response,
writes it to a temp file, executes it, and validates the output.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

PRIME_ENDPOINT = os.environ.get("PRIME_ENDPOINT", "http://localhost:7777")

def ask_prime(prompt, max_tokens=512):
    """Send a coding task to Prime and get the response."""
    body = json.dumps({
        "model": "test",
        "messages": [
            {"role": "system", "content": "You are GAIA, a sovereign AI. When asked to write code, output ONLY the Python code inside a ```python code block. No explanation before or after."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = Request(f"{PRIME_ENDPOINT}/v1/chat/completions", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]

def extract_code(response):
    """Extract Python code from markdown code blocks."""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
        return code.strip()
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
        return code.strip()
    return response.strip()

def run_code(code, timeout=10):
    """Execute Python code and return (success, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    finally:
        os.unlink(path)

# Test cases: (description, prompt, expected_in_output)
TESTS = [
    (
        "Fibonacci",
        "Write a Python function that computes the first 10 Fibonacci numbers and prints them as a list.",
        "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
    ),
    (
        "FizzBuzz",
        "Write a Python script that prints FizzBuzz for numbers 1-15. Print 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number.",
        "FizzBuzz"
    ),
    (
        "Sort + Dedupe",
        "Write a Python script that takes the list [3,1,4,1,5,9,2,6,5,3,5] removes duplicates, sorts it, and prints the result.",
        "[1, 2, 3, 4, 5, 6, 9]"
    ),
    (
        "JSON Parse",
        'Write a Python script that parses the JSON string \'{"name":"GAIA","tier":"prime","version":8}\' and prints each key-value pair on its own line as "key: value".',
        "name: GAIA"
    ),
    (
        "Palindrome Check",
        "Write a Python function that checks if a string is a palindrome (case-insensitive, ignoring spaces). Test it with 'A man a plan a canal Panama' and print True or False.",
        "True"
    ),
]

print(f"{'='*60}")
print(f"  GAIA Prime Code Generation Validator")
print(f"  Endpoint: {PRIME_ENDPOINT}")
print(f"  Tests: {len(TESTS)}")
print(f"{'='*60}\n")

passed = 0
failed = 0

for i, (name, prompt, expected) in enumerate(TESTS, 1):
    print(f"--- Test {i}/{len(TESTS)}: {name} ---")
    
    # Generate
    t0 = time.time()
    try:
        response = ask_prime(prompt)
    except Exception as e:
        print(f"  GENERATE FAILED: {e}")
        failed += 1
        continue
    gen_time = time.time() - t0
    
    # Extract code
    code = extract_code(response)
    print(f"  Generated: {len(code)} chars in {gen_time:.1f}s")
    
    # Execute
    success, stdout, stderr = run_code(code)
    
    if not success:
        print(f"  EXECUTE FAILED: {stderr[:200]}")
        print(f"  Code:\n{code[:300]}")
        failed += 1
        continue
    
    # Validate
    if expected in stdout:
        print(f"  PASS ✓ (output contains '{expected}')")
        print(f"  Output: {stdout[:150]}")
        passed += 1
    else:
        print(f"  FAIL ✗ (expected '{expected}' in output)")
        print(f"  Output: {stdout[:200]}")
        print(f"  Code:\n{code[:300]}")
        failed += 1

print(f"\n{'='*60}")
print(f"  Results: {passed}/{len(TESTS)} passed, {failed} failed")
print(f"{'='*60}")
