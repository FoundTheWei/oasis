#!/usr/bin/env python3
# Simple test script to verify LMStudio API functionality

import requests
import json
import time

def test_lmstudio_chat_completion():
    """Test if LMStudio responds correctly to a chat completion request"""
    url = "http://127.0.0.1:1234/v1/chat/completions"
    
    # This mirrors the format used in the ad_interaction.py file
    payload = {
        "model": "unsloth/deepseek-r1-distill-llama-8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "You are shown the following advertisement:\nHeadline: Summer Sale: 30% Off Everything!\nContent: Beat the heat with our biggest summer sale ever. Limited time only!\nCall to Action: Shop Now\n\nBased on your profile and interests, would you click on this ad? Please respond with either 'Yes, I would click this ad because...' or 'No, I would not click this ad because...' and explain your reasoning."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("Sending request to LMStudio API...")
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        elapsed_time = time.time() - start_time
        
        print(f"Request completed in {elapsed_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse content:")
            content = result['choices'][0]['message']['content']
            print(content)
            return True
        else:
            print(f"Error response: {response.text}")
            return False
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing LMStudio API connection...")
    success = test_lmstudio_chat_completion()
    
    if success:
        print("\nLMStudio API test completed successfully!")
    else:
        print("\nLMStudio API test failed. Please check your configuration.") 