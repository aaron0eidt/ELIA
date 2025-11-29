import requests
import json
import os

# API Configuration
API_CONFIG = {
    "api_key": os.environ.get("QWEN_API_KEY", "YOUR_API_KEY_HERE"),
    "api_endpoint": "https://chat-ai.academiccloud.de/v1",
    "model": "qwen2.5-vl-72b-instruct"
}

def check_usage():
    # Checks the current API usage by making a single call and checking the headers.
    headers = {
        "Authorization": f"Bearer {API_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": API_CONFIG["model"],
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{API_CONFIG['api_endpoint']}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print("=== API Usage Statistics ===")
        print(f"Status Code: {response.status_code}")
        print()
        
        # Extract rate limit information from the headers.
        rate_limit_info = {}
        for header, value in response.headers.items():
            if 'ratelimit' in header.lower():
                rate_limit_info[header] = value
        
        if rate_limit_info:
            print("Rate Limit Information:")
            for header, value in rate_limit_info.items():
                print(f"  {header}: {value}")
            
            print()
            
            # Calculate and display usage percentages.
            if 'X-RateLimit-Limit-Month' in rate_limit_info and 'X-RateLimit-Remaining-Month' in rate_limit_info:
                monthly_limit = int(rate_limit_info['X-RateLimit-Limit-Month'])
                monthly_remaining = int(rate_limit_info['X-RateLimit-Remaining-Month'])
                monthly_used = monthly_limit - monthly_remaining
                monthly_percentage = (monthly_used / monthly_limit) * 100
                
                print(f"Monthly Usage:")
                print(f"  Used: {monthly_used:,} requests")
                print(f"  Remaining: {monthly_remaining:,} requests")
                print(f"  Total Limit: {monthly_limit:,} requests")
                print(f"  Usage: {monthly_percentage:.1f}%")
            
            if 'X-RateLimit-Limit-Day' in rate_limit_info and 'X-RateLimit-Remaining-Day' in rate_limit_info:
                daily_limit = int(rate_limit_info['X-RateLimit-Limit-Day'])
                daily_remaining = int(rate_limit_info['X-RateLimit-Remaining-Day'])
                daily_used = daily_limit - daily_remaining
                daily_percentage = (daily_used / daily_limit) * 100
                
                print(f"\nDaily Usage:")
                print(f"  Used: {daily_used:,} requests")
                print(f"  Remaining: {daily_remaining:,} requests")
                print(f"  Total Limit: {daily_limit:,} requests")
                print(f"  Usage: {daily_percentage:.1f}%")
            
            if 'X-RateLimit-Limit-Hour' in rate_limit_info and 'X-RateLimit-Remaining-Hour' in rate_limit_info:
                hourly_limit = int(rate_limit_info['X-RateLimit-Limit-Hour'])
                hourly_remaining = int(rate_limit_info['X-RateLimit-Remaining-Hour'])
                hourly_used = hourly_limit - hourly_remaining
                hourly_percentage = (hourly_used / hourly_limit) * 100
                
                print(f"\nHourly Usage:")
                print(f"  Used: {hourly_used:,} requests")
                print(f"  Remaining: {hourly_remaining:,} requests")
                print(f"  Total Limit: {hourly_limit:,} requests")
                print(f"  Usage: {hourly_percentage:.1f}%")
                
                if hourly_remaining == 0:
                    reset_time = response.headers.get('RateLimit-Reset', 'Unknown')
                    print(f"  Reset in: {reset_time} seconds")
        
        if response.status_code == 429:
            print(f"\n⚠️  Rate limit exceeded!")
            print(f"Error: {response.text}")
        elif response.status_code == 200:
            print(f"\n✅ API call successful")
        else:
            print(f"\n❌ API call failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    check_usage() 