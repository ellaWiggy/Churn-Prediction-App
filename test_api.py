import requests
import json

api_url = "http://localhost:8000"

def test_search(client_id):
    print(f"\n--- Testing Search: Customer {client_id} ---")
    try:
        response = requests.get(f"{api_url}/customer/{client_id}", timeout=5)
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed with status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def test_prediction():
    print("\n--- Testing Prediction ---")
    input_data = {
        "dependent_count": 3,
        "education_level": "Graduate",
        "marital_status": "Married",
        "income_category": "$60K - $80K",
        "card_category": "Blue",
        "months_on_book": 39,
        "total_relationship_count": 5,
        "months_inactive_12_mon": 1,
        "contacts_count_12_mon": 2,
        "credit_limit": 5000.0,
        "total_revolving_bal": 1500.0,
        "total_amt_chng_q4_q1": 1.2,
        "total_trans_amt": 2000.0,
        "total_trans_ct": 50,
        "total_ct_chng_q4_q1": 0.8,
        "avg_utilization_ratio": 0.3
    }
    try:
        response = requests.post(f"{api_url}/predict", json=input_data, timeout=5)
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed with status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
def test_batch():
    try:
        response = requests.get(f"{api_url}/batch_predict", timeout=30)
        if response.status_code == 200:
            data = response.json()
            count = len(data.get("results", []))
            print(f"Success! Received {count} customer predictions.")
            print(json.dumps(data["results"][:3], indent=2))
        else:
            print(f"Failed: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Starting API Connectivity Tests...")
    
    test_search(713330558) 
    test_search(720244983)
    
    test_prediction()
    
    test_batch()