# Logout Function Debug Guide

## Issue: Logout Not Working Properly

### What I've Added

Enhanced the logout function with comprehensive logging to help diagnose the issue.

### How to Debug

1. **Open Browser Console** (F12 → Console tab)

2. **Click the Logout Button**

3. **Check Console Output** - You should see these logs in order:

```
Logout button event listener attached
Logout button clicked
Logout initiated...
Closing audio WebSocket...  (if audio was connected)
Stopping video stream...    (if video was running)
Clearing video analysis interval...
Destroying emotion chart...
Calling logout endpoint...
Logout successful: {code: 200, status: "success", ...}
Clearing localStorage...
Redirecting to login page...
```

### Possible Issues and What to Look For

#### Issue 1: Button Click Not Registered
**Console shows**: Nothing when you click logout
**Cause**: Event listener not attached
**Look for**: Error message "Logout button not found!"
**Solution**: Check if `logout-btn` ID exists in HTML

#### Issue 2: Logout Endpoint Fails
**Console shows**: 
```
Logout endpoint returned error: 500 ...
```
**Cause**: Backend error
**Solution**: Check backend logs for errors

#### Issue 3: Network Error
**Console shows**:
```
Logout API call failed: TypeError: Failed to fetch
```
**Cause**: Backend not running or CORS issue
**Solution**: Ensure backend is running on port 8000

#### Issue 4: Token Not Found
**Console shows**:
```
No token found in localStorage
```
**Cause**: Token was already cleared or never set
**Solution**: Check login process

#### Issue 5: Redirect Fails
**Console shows**: All logs but page doesn't redirect
**Cause**: JavaScript error after redirect call
**Solution**: Check for errors after "Redirecting to login page..."

#### Issue 6: Resources Not Cleaned Up
**Console shows**: Logs but audio/video still running
**Cause**: Cleanup code not executing
**Solution**: Check if audio/video variables are null

### Testing Steps

1. **Test Basic Logout**
   ```
   - Login
   - Immediately click Logout
   - Check console logs
   - Should redirect to login page
   ```

2. **Test with Audio Active**
   ```
   - Login
   - Navigate to Audio Monitoring (audio connects)
   - Click Logout
   - Should see "Closing audio WebSocket..."
   - Should redirect to login page
   ```

3. **Test with Video Active**
   ```
   - Login
   - Navigate to Video Monitoring
   - Click "Start Video"
   - Click Logout
   - Should see "Stopping video stream..."
   - Should redirect to login page
   ```

4. **Test with Both Active**
   ```
   - Login
   - Start video
   - Navigate to audio (auto-connects)
   - Click Logout
   - Should see both cleanup logs
   - Should redirect to login page
   ```

### What Each Log Means

| Log Message | What It Means |
|------------|---------------|
| `Logout button event listener attached` | Button is ready to receive clicks |
| `Logout button clicked` | User clicked the button |
| `Logout initiated...` | Logout function started |
| `Closing audio WebSocket...` | Disconnecting audio stream |
| `Stopping video stream...` | Stopping webcam |
| `Clearing video analysis interval...` | Stopping frame analysis |
| `Destroying emotion chart...` | Cleaning up Chart.js |
| `Calling logout endpoint...` | Sending request to backend |
| `Logout successful: {...}` | Backend confirmed logout |
| `Clearing localStorage...` | Removing token and username |
| `Redirecting to login page...` | About to redirect |

### Common Problems

#### Problem: "Logout button not found!"
**Fix**: Check `dashboard.html` for `<button id="logout-btn">`

#### Problem: Logout works but can still access dashboard
**Fix**: Backend token blocklist not working
**Check**: 
```javascript
// After logout, try accessing dashboard again
// Should redirect to login immediately
```

#### Problem: Page doesn't redirect
**Fix**: Check for JavaScript errors after redirect call
**Try**: Hard refresh (Ctrl+Shift+R)

#### Problem: Audio/Video still running after logout
**Fix**: Check if cleanup code is executing
**Verify**: Console shows cleanup logs

### Manual Test in Console

If logout button doesn't work, try calling the function directly:

```javascript
// Open browser console (F12)
// Type this and press Enter:
handleLogout();
```

This will:
- Show if the function exists
- Show any errors
- Execute the logout process

### Backend Verification

Check if token is blocklisted:

1. **Before Logout**: Note your token
   ```javascript
   console.log(localStorage.getItem('token'));
   ```

2. **After Logout**: Try using the old token
   ```javascript
   // Should fail with 401 Unauthorized
   fetch('http://localhost:8000/detections/history', {
       headers: { 'Authorization': 'Bearer YOUR_OLD_TOKEN' }
   }).then(r => console.log(r.status));
   ```

### Files Modified

1. **`frontend/js/dashboard.js`**
   - Added logging to `handleLogout()` function
   - Added logging to logout button event listener
   - Added null check for logout button
   - Added video element cleanup

### Expected Behavior

**When Working Correctly:**
1. Click logout button
2. See all cleanup logs in console
3. Page redirects to `index.html`
4. localStorage is cleared
5. Old token is blocklisted
6. Cannot access dashboard without logging in again

### Still Not Working?

If you've checked all the above and it still doesn't work:

1. **Share Console Output**: Copy all console logs when clicking logout
2. **Check Backend Logs**: Look for errors in backend terminal
3. **Check Network Tab**: F12 → Network → Click logout → Check if POST to `/logout` appears
4. **Try Different Browser**: Test in Chrome, Edge, or Firefox
5. **Clear Browser Cache**: Hard refresh (Ctrl+Shift+R)

### Quick Fix

If logout is completely broken, you can manually clear everything:

```javascript
// Open console (F12) and run:
localStorage.clear();
window.location.href = 'index.html';
```

This will:
- Clear all localStorage
- Redirect to login page
- Force a fresh start

---

## Summary

The logout function now has comprehensive logging. Open the browser console and click logout to see exactly what's happening at each step. This will help identify where the issue is occurring.
