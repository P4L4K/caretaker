"""
Quick API Test Script
Run this after starting the server to verify all endpoints work
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_signup():
    """Test signup endpoint"""
    print("\n2. Testing Signup...")
    data = {
        "username": "test_user",
        "email": "test@example.com",
        "password": "TestPass123",
        "phone_number": "1234567890",
        "full_name": "Test User",
        "care_recipients": [
            {
                "full_name": "Test Recipient",
                "email": "recipient@example.com",
                "phone_number": "9876543210",
                "age": 70,
                "gender": "Female",
                "respiratory_condition_status": False
            }
        ]
    }
    try:
        response = requests.post(f"{BASE_URL}/signup", json=data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code in [200, 400]  # 400 if user exists
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_login():
    """Test login endpoint"""
    print("\n3. Testing Login...")
    data = {
        "username": "test_user",
        "password": "TestPass123"
    }
    try:
        response = requests.post(f"{BASE_URL}/login", json=data)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Token received: {result.get('access_token', 'N/A')[:50]}...")
        return response.status_code == 200, result.get('access_token')
    except Exception as e:
        print(f"   Error: {e}")
        return False, None

def test_cough_detections(token):
    """Test cough detections endpoint"""
    print("\n4. Testing Cough Detections Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/cough/detections")
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Detections found: {len(result.get('items', []))}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_docs():
    """Test API documentation"""
    print("\n5. Testing API Documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"   Status: {response.status_code}")
        print(f"   Docs available: {response.status_code == 200}")
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False

def main():
    print("=" * 60)
    print("CareTaker AI - API Test Suite")
    print("=" * 60)
    print("\nMake sure the server is running: python main.py")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Signup", test_signup()))
    success, token = test_login()
    results.append(("Login", success))
    
    if token:
        results.append(("Cough Detections", test_cough_detections(token)))
    else:
        results.append(("Cough Detections", False))
    
    results.append(("API Docs", test_docs()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Server is working correctly.")
        print("\nNext steps:")
        print("1. Connect to WebSocket: ws://localhost:8000/ws/audio?token=YOUR_TOKEN")
        print("2. Open video stream: http://localhost:8000/video/stream")
        print("3. View API docs: http://localhost:8000/docs")
    else:
        print("\n⚠️ Some tests failed. Check the server logs for errors.")

if __name__ == "__main__":
    main()
