# LLM Integration Setup Guide

Your CRM chatbot now supports enhanced AI responses using Large Language Models (LLM)! This guide will help you configure OpenAI or Anthropic Claude to power intelligent, context-aware conversations.

## 🚀 Quick Setup

### 1. Get an API Key

#### Option A: OpenAI (Recommended)

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Navigate to "API Keys" section
4. Click "Create new secret key"
5. Copy your API key (starts with `sk-`)

#### Option B: Anthropic Claude (Alternative)

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account or sign in
3. Generate an API key
4. Copy your API key

### 2. Configure Environment Variable

Add your API key to your environment:

**For local development:**

```bash
# Create a .env file in your project root
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

**For production deployment:**
Set the environment variable in your hosting platform:

- **Vercel**: Project Settings → Environment Variables
- **Netlify**: Site Settings → Environment Variables
- **Railway/Render**: Environment tab

### 3. Restart Your Application

After adding the API key, restart your development server:

```bash
npm run dev
```

## 🎯 Features Enabled

With LLM integration, your chatbot can now:

- **Natural Language Understanding**: Process complex, conversational queries
- **Context-Aware Responses**: Understand conversation history and context
- **Personalized Insights**: Provide tailored recommendations based on your CRM data
- **Dynamic Analysis**: Generate insights beyond predefined rules
- **Smart Follow-ups**: Suggest relevant next actions based on conversation flow

## 💡 Example Conversations

### Before LLM (Keyword-based):

**User**: "leads"  
**Bot**: Shows generic lead list

### After LLM (Natural Language):

**User**: "I'm struggling to prioritize my leads this week. Can you help me figure out which ones I should focus on first?"  
**Bot**: Analyzes your actual lead data, considers scores, values, and status to provide personalized prioritization advice with specific recommendations.

## 🔧 Configuration Options

### Environment Variables

| Variable            | Required | Description                   | Example      |
| ------------------- | -------- | ----------------------------- | ------------ |
| `OPENAI_API_KEY`    | Optional | OpenAI API key for GPT models | `sk-...`     |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key for Claude  | `sk-ant-...` |

### Fallback Behavior

If no API key is configured:

- ✅ Chatbot continues to work with rule-based responses
- ⚠️ Limited to keyword matching
- 📝 Status indicator shows "Standard Mode"

If API key is configured:

- 🤖 Enhanced AI responses enabled
- 🧠 Natural language processing
- 📊 Context-aware analysis
- ✨ Status indicator shows "AI Enhanced"

## 💰 Cost Considerations

### OpenAI Pricing (GPT-3.5-turbo)

- **Input**: ~$0.0005 per 1K tokens
- **Output**: ~$0.0015 per 1K tokens
- **Typical chat response**: $0.001 - $0.005 per interaction

### Tips to Minimize Costs

1. **Start Small**: Test with a few users first
2. **Monitor Usage**: Check OpenAI dashboard regularly
3. **Set Limits**: Configure billing alerts
4. **Optimize Prompts**: Keep context relevant and concise

## 🔒 Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive data
3. **Rotate keys regularly** (monthly recommended)
4. **Monitor usage** for unusual activity
5. **Set spending limits** in your LLM provider dashboard

## 🚨 Troubleshooting

### Chatbot shows "Standard Mode"

- ✅ Check if `OPENAI_API_KEY` is set correctly
- ✅ Verify API key format (starts with `sk-`)
- ✅ Restart your application after adding the key
- ✅ Check browser console for error messages

### API errors or failed responses

- ✅ Verify your OpenAI account has available credits
- ✅ Check API key permissions and usage limits
- ✅ Monitor OpenAI status page for service issues

### High API costs

- ✅ Review conversation length (affects token usage)
- ✅ Set up billing alerts in OpenAI dashboard
- ✅ Consider using GPT-3.5 instead of GPT-4 for cost optimization

## 📞 Support

If you need help with LLM integration:

1. Check the browser console for error messages
2. Verify your API key configuration
3. Test the connection using the admin panel
4. Review OpenAI/Anthropic documentation for API issues

---

**Ready to experience intelligent conversations with your CRM data!** 🎉
