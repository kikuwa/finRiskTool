import requests
import time
import os

BASE_URL = "http://localhost:5005"

def test_endpoint(name, url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"[PASS] {name} ({url}) is accessible.")
            return True
        else:
            print(f"[FAIL] {name} ({url}) returned status code {response.status_code}.")
            return False
    except Exception as e:
        print(f"[FAIL] {name} ({url}) could not be reached. Error: {e}")
        return False

def test_post_endpoint(name, url):
    try:
        print(f"Testing {name}...")
        start_time = time.time()
        # Set a reasonable timeout. PU bagging might be slow.
        response = requests.post(url, timeout=300) 
        duration = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"[PASS] {name} executed successfully in {duration:.2f}s.")
                return True
            else:
                print(f"[FAIL] {name} executed but returned failure: {data.get('error')}")
                # Print stderr if available for debugging
                if 'stderr' in data:
                    print(f"Stderr: {data['stderr']}")
                return False
        else:
            print(f"[FAIL] {name} returned status code {response.status_code}.")
            return False
    except requests.exceptions.Timeout:
        print(f"[WARN] {name} timed out after 300s. It might still be running in background or too slow.")
        return False
    except Exception as e:
        print(f"[FAIL] {name} error: {e}")
        return False

def main():
    print("Starting Project Health Check...")
    
    # 1. Page Accessibility Tests
    pages = [
        ("Home Page", "/"),
        ("PU Learning Page", "/data_tool/pu_bagging"),
        ("AutoML Page", "/data_tool/ensemble_feature_selection"),
        ("Risk CoT Generator", "/risk_cot/generator"),
        ("Risk CoT Inference", "/risk_cot/inference"),
        ("Risk CoT Inspector", "/risk_cot/inspector"),
    ]
    
    all_pages_pass = True
    for name, path in pages:
        if not test_endpoint(name, BASE_URL + path):
            all_pages_pass = False
    
    if not all_pages_pass:
        print("\nSome pages are not accessible. Please check the server logs.")
        return

    # 2. Functional Tests (Model Execution)
    # Note: These are heavy operations.
    
    print("\nStarting Functional Tests (Model Execution)...")
    
    # Test PU Bagging and AutoML
    # We assume data/train.csv exists (we copied it)
    if os.path.exists("data/train.csv"):
        print("data/train.csv found. Proceeding with functional tests.")
        
        # Test PU Bagging
        # This might take a while
        test_post_endpoint("PU Bagging Model", BASE_URL + "/data_tool/run_model")

        # Test AutoML
        test_post_endpoint("AutoML Feature Selection", BASE_URL + "/data_tool/run_model_feature_selection")
    else:
        print("[SKIP] data/train.csv not found. Skipping functional tests.")

if __name__ == "__main__":
    main()
