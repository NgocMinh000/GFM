#!/bin/bash
# test_yescale_setup.sh - Verify YEScale API configuration

echo "=================================="
echo "YEScale API Configuration Check"
echo "=================================="
echo ""

# Check YESCALE_API_BASE_URL
if [ -z "$YESCALE_API_BASE_URL" ]; then
    echo "❌ YESCALE_API_BASE_URL is NOT set"
    echo "   Please set it:"
    echo "   export YESCALE_API_BASE_URL=\"https://api.yescale.io/v1/chat/completions\""
    MISSING_URL=1
else
    echo "✅ YESCALE_API_BASE_URL is set"
    echo "   Value: $YESCALE_API_BASE_URL"
fi

echo ""

# Check YESCALE_API_KEY or OPENAI_API_KEY
if [ -z "$YESCALE_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ YESCALE_API_KEY (or OPENAI_API_KEY) is NOT set"
    echo "   Please set one of them:"
    echo "   export YESCALE_API_KEY=\"sk-xxxxx\""
    echo "   OR"
    echo "   export OPENAI_API_KEY=\"sk-xxxxx\""
    MISSING_KEY=1
else
    if [ -n "$YESCALE_API_KEY" ]; then
        echo "✅ YESCALE_API_KEY is set"
        echo "   Value: ${YESCALE_API_KEY:0:10}... (hidden)"
    else
        echo "✅ OPENAI_API_KEY is set (will be used as YEScale key)"
        echo "   Value: ${OPENAI_API_KEY:0:10}... (hidden)"
    fi
fi

echo ""
echo "=================================="

if [ -n "$MISSING_URL" ] || [ -n "$MISSING_KEY" ]; then
    echo "⚠️  Configuration INCOMPLETE"
    echo ""
    echo "To fix, add these to your ~/.bashrc or run in terminal:"
    echo ""
    echo "export YESCALE_API_BASE_URL=\"https://api.yescale.io/v1/chat/completions\""
    echo "export YESCALE_API_KEY=\"your-api-key-here\""
    echo ""
    echo "Then reload: source ~/.bashrc"
    exit 1
else
    echo "✅ Configuration COMPLETE"
    echo ""
    echo "YEScale API is properly configured."
    echo "Stage 0 LLM-based relationship inference will work."
    exit 0
fi
