#!/usr/bin/env python3
"""
Test script để kiểm tra kết nối với YEScale API
Sử dụng OpenAI Python SDK với custom base_url
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_with_openai_sdk():
    """Test kết nối YEScale bằng OpenAI Python SDK"""
    print("=" * 60)
    print("Test 1: Sử dụng OpenAI SDK với YEScale base_url")
    print("=" * 60)

    try:
        from openai import OpenAI

        # Get credentials from environment
        api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("YESCALE_API_BASE_URL")

        if not api_key:
            print("❌ ERROR: YESCALE_API_KEY hoặc OPENAI_API_KEY không được set!")
            return False

        print(f"✓ API Key: {api_key[:10]}...")
        print(f"✓ Base URL: {base_url or 'None (sẽ dùng OpenAI default)'}")

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            print(f"✓ Sẽ gọi đến: {base_url}/chat/completions")

        client = OpenAI(**client_kwargs)

        # Test request
        print("\n🔄 Đang gửi test request...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello! Reply with just 'Hi' in one word."}
            ],
            max_tokens=10,
            temperature=0.0
        )

        print(f"✅ SUCCESS! Response:")
        print(f"   - ID: {response.id}")
        print(f"   - Model: {response.model}")
        print(f"   - Content: {response.choices[0].message.content}")
        print(f"   - Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_requests():
    """Test kết nối YEScale bằng raw requests (theo docs YEScale)"""
    print("\n" + "=" * 60)
    print("Test 2: Sử dụng requests library (raw HTTP)")
    print("=" * 60)

    try:
        import requests
        import json

        # Get credentials
        api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("YESCALE_API_BASE_URL")

        if not api_key:
            print("❌ ERROR: API key không được set!")
            return False

        if not base_url:
            print("⚠️  WARNING: YESCALE_API_BASE_URL không được set, skip test này")
            return False

        # Build full URL
        url = f"{base_url}/chat/completions"
        print(f"✓ Full URL: {url}")

        # Prepare request
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Reply with just 'Hi' in one word."
                }
            ],
            "max_tokens": 10
        }

        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        print("🔄 Đang gửi raw HTTP request...")
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Response:")
            print(f"   - ID: {data['id']}")
            print(f"   - Content: {data['choices'][0]['message']['content']}")
            print(f"   - Tokens: {data['usage']['total_tokens']}")
            return True
        else:
            print(f"❌ ERROR: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chatgpt_class():
    """Test kết nối bằng ChatGPT class từ repo"""
    print("\n" + "=" * 60)
    print("Test 3: Sử dụng ChatGPT class từ gfmrag.llms")
    print("=" * 60)

    try:
        # Add gfmrag to path
        sys.path.insert(0, os.path.dirname(__file__))
        from gfmrag.llms import ChatGPT

        api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("YESCALE_API_BASE_URL")

        if not api_key:
            print("❌ ERROR: API key không được set!")
            return False

        print(f"✓ API Key: {api_key[:10]}...")
        print(f"✓ Base URL: {base_url or 'None (OpenAI default)'}")

        # Initialize ChatGPT với YEScale params
        print("🔄 Khởi tạo ChatGPT class...")
        llm = ChatGPT(
            model_name_or_path="gpt-4o-mini",
            api_key=api_key,
            base_url=base_url
        )

        # Test generate
        print("🔄 Đang gọi generate_sentence()...")
        response = llm.generate_sentence("Hello! Reply with just 'Hi' in one word.")

        print(f"✅ SUCCESS! Response:")
        print(f"   - Content: {response}")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_langchain_model():
    """Test kết nối bằng LangChain model từ repo"""
    print("\n" + "=" * 60)
    print("Test 4: Sử dụng LangChain init_langchain_model()")
    print("=" * 60)

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from gfmrag.kg_construction.langchain_util import init_langchain_model

        api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("YESCALE_API_BASE_URL")

        if not api_key:
            print("❌ ERROR: API key không được set!")
            return False

        print(f"✓ API Key: {api_key[:10]}...")
        print(f"✓ Base URL: {base_url or 'None'}")

        # Initialize model
        print("🔄 Khởi tạo LangChain model...")
        llm = init_langchain_model("openai", "gpt-4o-mini", temperature=0.0)

        # Test invoke
        print("🔄 Đang gọi llm.invoke()...")
        response = llm.invoke("Hello! Reply with just 'Hi' in one word.")

        print(f"✅ SUCCESS! Response:")
        print(f"   - Content: {response.content}")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🔧" * 30)
    print("YEScale API Connection Test Suite")
    print("🔧" * 30 + "\n")

    # Check environment variables
    print("Environment Variables:")
    print(f"  YESCALE_API_KEY: {'✓ Set' if os.environ.get('YESCALE_API_KEY') else '✗ Not set'}")
    print(f"  YESCALE_API_BASE_URL: {os.environ.get('YESCALE_API_BASE_URL') or '✗ Not set'}")
    print(f"  OPENAI_API_KEY: {'✓ Set (fallback)' if os.environ.get('OPENAI_API_KEY') else '✗ Not set'}")
    print()

    # Run tests
    results = []

    results.append(("OpenAI SDK", test_with_openai_sdk()))
    results.append(("Raw Requests", test_with_requests()))
    results.append(("ChatGPT Class", test_chatgpt_class()))
    results.append(("LangChain Model", test_langchain_model()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! YEScale integration is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
