import requests

API_URL = "http://localhost:8000/api/v1"

def test_create_student_empty_email():
    data = {
        "student_id": "TEST001",
        "full_name": "Test User",
        "email": "",
        "phone": ""
    }
    response = requests.post(f"{API_URL}/students/", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_create_student_empty_email()
