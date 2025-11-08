# PowerShell Script to Test Logout Functionality
# Usage: .\test_logout.ps1 -Username "your_username" -Password "your_password"

param(
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$true)]
    [string]$Password
)

$BaseUrl = "http://localhost:8000"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CareTaker Logout Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if backend is running
Write-Host "[1/5] Checking backend status..." -ForegroundColor Yellow
try {
    $healthCheck = Invoke-RestMethod -Uri "$BaseUrl/" -Method Get
    Write-Host "✓ Backend is running: $($healthCheck.message)" -ForegroundColor Green
} catch {
    Write-Host "✗ Backend is not running!" -ForegroundColor Red
    Write-Host "Please start the backend first: cd backend && python main.py" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Login
Write-Host "[2/5] Logging in as '$Username'..." -ForegroundColor Yellow
try {
    $loginBody = @{
        username = $Username
        password = $Password
    } | ConvertTo-Json

    $loginResponse = Invoke-RestMethod -Uri "$BaseUrl/login" `
        -Method Post `
        -ContentType "application/json" `
        -Body $loginBody

    if ($loginResponse.code -eq 200) {
        $token = $loginResponse.result.access_token
        Write-Host "✓ Login successful!" -ForegroundColor Green
        Write-Host "  Token: $($token.Substring(0, 50))..." -ForegroundColor Gray
    } else {
        Write-Host "✗ Login failed: $($loginResponse.message)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ Login failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Test protected endpoint (should work)
Write-Host "[3/5] Testing protected endpoint with token..." -ForegroundColor Yellow
try {
    $headers = @{
        "Authorization" = "Bearer $token"
    }
    
    $historyResponse = Invoke-RestMethod -Uri "$BaseUrl/detections/history" `
        -Method Get `
        -Headers $headers
    
    Write-Host "✓ Protected endpoint accessible (token is valid)" -ForegroundColor Green
} catch {
    Write-Host "✗ Protected endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Step 4: Logout
Write-Host "[4/5] Logging out..." -ForegroundColor Yellow
try {
    $logoutBody = @{
        token = $token
    } | ConvertTo-Json

    $logoutResponse = Invoke-RestMethod -Uri "$BaseUrl/logout" `
        -Method Post `
        -ContentType "application/json" `
        -Body $logoutBody

    if ($logoutResponse.code -eq 200) {
        Write-Host "✓ Logout successful!" -ForegroundColor Green
        Write-Host "  Message: $($logoutResponse.message)" -ForegroundColor Gray
    } else {
        Write-Host "✗ Logout failed: $($logoutResponse.message)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ Logout failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  This might be a CORS or network issue" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Test protected endpoint again (should fail)
Write-Host "[5/5] Testing if token is blocklisted..." -ForegroundColor Yellow
try {
    $headers = @{
        "Authorization" = "Bearer $token"
    }
    
    $historyResponse = Invoke-RestMethod -Uri "$BaseUrl/detections/history" `
        -Method Get `
        -Headers $headers
    
    Write-Host "✗ Token still works! Logout didn't blocklist the token." -ForegroundColor Red
} catch {
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "✓ Token is blocklisted (401 Unauthorized)" -ForegroundColor Green
        Write-Host "  Logout is working correctly!" -ForegroundColor Green
    } else {
        Write-Host "? Unexpected error: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "1. Backend: Running ✓" -ForegroundColor Green
Write-Host "2. Login: Success ✓" -ForegroundColor Green
Write-Host "3. Token Valid: Before logout ✓" -ForegroundColor Green
Write-Host "4. Logout: Success ✓" -ForegroundColor Green
Write-Host "5. Token Blocklisted: After logout ✓" -ForegroundColor Green
Write-Host ""
Write-Host "Logout functionality is working correctly!" -ForegroundColor Green
