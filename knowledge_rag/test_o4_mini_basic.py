#!/usr/bin/env python3
"""
Basic test to see if o4-mini responds to simple prompts
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_basic_o4_mini():
    """Test o4-mini with the simplest possible prompt"""
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    print("🧪 Testing o4-mini with basic prompts...")
    
    # Test 1: Simplest possible
    print("\n--- Test 1: Simplest possible ---")
    try:
        response = client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_completion_tokens=100,
            reasoning_effort="low"
        )
        
        content = response.choices[0].message.content
        print(f"✅ Response: '{content}'")
        print(f"📊 Length: {len(content) if content else 0} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Simple JSON request (no format specification)
    print("\n--- Test 2: Simple JSON request (no format) ---")
    try:
        response = client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "user", "content": "Return a simple JSON object with your name"}
            ],
            max_completion_tokens=100,
            reasoning_effort="low"
        )
        
        content = response.choices[0].message.content
        print(f"✅ Response: '{content}'")
        print(f"📊 Length: {len(content) if content else 0} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: With JSON format specification
    print("\n--- Test 3: With JSON format specification ---")
    try:
        response = client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "user", "content": "Return a JSON object with one field called 'message' containing 'hello'"}
            ],
            max_completion_tokens=100,
            reasoning_effort="low",
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        print(f"✅ Response: '{content}'")
        print(f"📊 Length: {len(content) if content else 0} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Entity extraction (very simple)
    print("\n--- Test 4: Very simple entity extraction ---")
    try:
        response = client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "user", "content": "Find entities in this text: 'John works at Google'. Return JSON with entities array."}
            ],
            max_completion_tokens=200,
            reasoning_effort="low",
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        print(f"✅ Response: '{content}'")
        print(f"📊 Length: {len(content) if content else 0} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_basic_o4_mini() 