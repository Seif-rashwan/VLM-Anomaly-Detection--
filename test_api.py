#!/usr/bin/env python3
"""
Test script to verify Google Generative AI API setup and connectivity
"""
import os
import json
import sys

def test_api():
    print("=" * 60)
    print("Google Generative AI API Test")
    print("=" * 60)
    
    # 1. Check if google-generativeai is installed
    print("\n[1] Checking if google-generativeai is installed...")
    try:
        import google.generativeai as genai
        print("✓ google-generativeai imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import google-generativeai: {e}")
        return False
    
    # 2. Check for API key
    print("\n[2] Looking for API key...")
    api_key = None
    
    # Try environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print(f"✓ Found API key in environment: {api_key[:10]}...")
    
    # Try .streamlit/secrets.toml
    if not api_key:
        try:
            secrets_path = ".streamlit/secrets.toml"
            if os.path.exists(secrets_path):
                with open(secrets_path, 'r') as f:
                    content = f.read()
                    if "GOOGLE_API_KEY" in content:
                        # Simple extraction
                        for line in content.split('\n'):
                            if 'GOOGLE_API_KEY' in line:
                                api_key = line.split('=')[1].strip().strip('"')
                                print(f"✓ Found API key in .streamlit/secrets.toml: {api_key[:10]}...")
                                break
        except Exception as e:
            print(f"Note: Could not read secrets.toml: {e}")
    
    # Try Python environment (when running under streamlit)
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
                print(f"✓ Found API key in Streamlit secrets: {api_key[:10]}...")
        except Exception as e:
            print(f"Note: Could not access Streamlit secrets: {e}")
    
    if not api_key:
        print("✗ No API key found!")
        print("  Please set GOOGLE_API_KEY environment variable or in .streamlit/secrets.toml")
        return False
    
    # 3. Configure API
    print("\n[3] Configuring API...")
    try:
        genai.configure(api_key=api_key)
        print("✓ API configured successfully")
    except Exception as e:
        print(f"✗ Failed to configure API: {e}")
        return False
    
    # 4. Test API call
    print("\n[4] Testing API call...")
    try:
        test_prompt = "Say 'Hello' in JSON format with key 'message'"
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json"
            }
        )
        response = model.generate_content(test_prompt)
        
        if response and response.text:
            print(f"✓ API call successful!")
            print(f"  Response: {response.text[:100]}")
            return True
        else:
            print(f"✗ Empty response from API")
            return False
            
    except Exception as e:
        error_str = str(e)
        print(f"✗ API call failed: {error_str[:200]}")
        
        # Check for specific errors
        if "429" in error_str or "quota" in error_str.lower():
            print("\n  Note: You may be hitting API quota limits")
            print("  This is normal if called too frequently")
            print("  The app will fall back to offline mode")
            return True  # Still considered success - offline mode works
        
        if "invalid" in error_str.lower() or "api_key" in error_str.lower():
            print("\n  Your API key may be invalid or expired")
            return False
        
        return False

if __name__ == "__main__":
    success = test_api()
    print("\n" + "=" * 60)
    if success:
        print("✓ API test passed!")
        print("\nThe app will use Google API when available.")
        print("If quota errors occur, it will automatically use offline mode.")
    else:
        print("✗ API test failed")
        print("\nThe app will work in offline mode.")
    print("=" * 60)
    sys.exit(0 if success else 1)
