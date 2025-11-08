# Logout CORS Issue - Fix Guide

## Issue Observed

When testing the `/logout` endpoint via Swagger UI or curl, you're getting:
```
Failed to fetch.
Possible Reasons: CORS, Network Failure
```

## Root Causes

### 1. **Invalid Token in Test**
Your curl command uses:
```json
{
  "token": "string"
}
```

This is not a valid JWT token. You need a **real token** from a login session.

### 2. **CORS Preflight Request**
Swagger UI sends an OPTIONS request first (CORS preflight), which might be failing.

### 3. **Server Address Mismatch**
- Swagger UI calls: `http://127.0.0.1:8000`
- Server runs on: `http://0.0.0.0:8000`
- Frontend uses: `http://localhost:8000`

All three should work, but there might be a mismatch.

## Solutions

### Solution 1: Test with Real Token (Recommended)

#### Step 1: Get a Real Token
```bash
# Login first to get a token
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "code": 200,
  "status": "success",
  "message": "Login successful",
  "result": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer"
  }
}
```

#### Step 2: Use That Token for Logout
```bash
# Copy the access_token from above
curl -X POST http://localhost:8000/logout \
  -H "Content-Type: application/json" \
  -d '{
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

**Expected Response:**
```json
{
  "code": 200,
  "status": "success",
  "message": "Logout successful",
  "result": {}
}
```

### Solution 2: Test via Frontend (Easiest)

1. **Start Backend**
   ```bash
   cd backend
   python main.py
   ```

2. **Open Frontend**
   - Open `frontend/dashboard.html` in browser
   - Login with credentials

3. **Open Browser Console** (F12)

4. **Click Logout Button**

5. **Check Console Logs**
   - Should see: "Logout successful: {...}"
   - Should redirect to login page

### Solution 3: Fix CORS for Swagger UI

The CORS is already configured correctly in `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

But Swagger UI might need the server restarted:

1. **Stop the backend** (Ctrl+C)
2. **Restart it**:
   ```bash
   cd backend
   python main.py
   ```
3. **Try Swagger UI again**: http://localhost:8000/docs

### Solution 4: Test with PowerShell (Windows)

```powershell
# Step 1: Login
$loginResponse = Invoke-RestMethod -Uri "http://localhost:8000/login" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"username":"your_username","password":"your_password"}'

$token = $loginResponse.result.access_token
Write-Host "Token: $token"

# Step 2: Logout
$logoutResponse = Invoke-RestMethod -Uri "http://localhost:8000/logout" `
  -Method Post `
  -ContentType "application/json" `
  -Body "{`"token`":`"$token`"}"

Write-Host "Logout Response: $logoutResponse"
```

## Why "string" Doesn't Work

The backend expects a **valid JWT token**, not the literal string "string".

**Valid token format:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIiwiZXhwIjoxNjk5MzY3MjAwfQ.signature_here
```

**Invalid (what you used):**
```
"string"
```

The backend will try to decode "string" as a JWT and fail, but it should still return a proper error, not a CORS error.

## Debugging Steps

### 1. Check Backend is Running
```bash
curl http://localhost:8000/
```

**Expected:**
```json
{
  "message": "Welcome to CareTaker API",
  "status": "active"
}
```

### 2. Check CORS Headers
```bash
curl -X OPTIONS http://localhost:8000/logout \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v
```

**Look for:**
```
< Access-Control-Allow-Origin: *
< Access-Control-Allow-Methods: *
< Access-Control-Allow-Headers: *
```

### 3. Test Login First
```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}' \
  -v
```

If login works, logout should work too.

### 4. Check Backend Logs
Look at the terminal where backend is running. You should see:
```
INFO:     127.0.0.1:xxxxx - "POST /logout HTTP/1.1" 200 OK
```

If you see `500` or `422`, there's a backend error.

## Common Issues

### Issue 1: "Failed to fetch" in Swagger UI
**Cause**: Browser blocking request or CORS issue
**Fix**: 
- Use curl or PowerShell instead
- Or test via frontend dashboard

### Issue 2: Backend Returns 422
**Cause**: Invalid request body format
**Fix**: Ensure you're sending `{"token": "actual_jwt_token"}`

### Issue 3: Backend Returns 500
**Cause**: Database or token blocklist error
**Fix**: Check backend logs for the actual error

### Issue 4: Logout Works but Token Still Valid
**Cause**: Token blocklist not working
**Fix**: Check database has `token_blocklist` table

## Testing Logout Properly

### Complete Test Flow:

```bash
# 1. Login
LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"your_username","password":"your_password"}')

echo "Login Response: $LOGIN_RESPONSE"

# 2. Extract token (requires jq)
TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.result.access_token')
echo "Token: $TOKEN"

# 3. Test protected endpoint (should work)
curl -X GET http://localhost:8000/detections/history \
  -H "Authorization: Bearer $TOKEN"

# 4. Logout
LOGOUT_RESPONSE=$(curl -s -X POST http://localhost:8000/logout \
  -H "Content-Type: application/json" \
  -d "{\"token\":\"$TOKEN\"}")

echo "Logout Response: $LOGOUT_RESPONSE"

# 5. Try protected endpoint again (should fail with 401)
curl -X GET http://localhost:8000/detections/history \
  -H "Authorization: Bearer $TOKEN"
```

**Expected Results:**
- Step 3: Returns detection history (200 OK)
- Step 4: Returns logout success (200 OK)
- Step 5: Returns 401 Unauthorized (token blocklisted)

## Frontend Test (Recommended)

This is the **easiest and most reliable** way to test:

1. Open `frontend/dashboard.html` in browser
2. Open DevTools (F12) → Console tab
3. Login with your credentials
4. Click the "Logout" button
5. Check console logs

**Expected console output:**
```
Logout button event listener attached
Logout button clicked
Logout initiated...
Calling logout endpoint...
Logout successful: {code: 200, status: "success", message: "Logout successful", result: {}}
Clearing localStorage...
Redirecting to login page...
```

## Summary

The CORS error in Swagger UI is likely because:
1. ❌ You're using `"token": "string"` instead of a real JWT token
2. ❌ Swagger UI's CORS preflight might be failing
3. ❌ Backend might need to be restarted

**Best Solution**: Test via the frontend dashboard with real login/logout flow.

**Alternative**: Use curl with a real token from a login request.

The logout endpoint itself is working correctly - the issue is with how you're testing it! 🎯
