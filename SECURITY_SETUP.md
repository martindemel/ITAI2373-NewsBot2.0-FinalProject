# NewsBot 2.0 Security Setup Guide

## 🔐 API Key Configuration

### CRITICAL: Before Running the Application

Your NewsBot 2.0 system requires an OpenAI API key to function properly. Follow these steps to set it up securely:

### Step 1: Get Your OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in to your account
3. Create a new API key
4. Copy the key (starts with `sk-proj-` or `sk-`)

### Step 2: Configure Environment Variables

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Replace the placeholder:**
   ```bash
   # Change this line:
   OPENAI_API_KEY=your-openai-api-key-here
   
   # To your actual key:
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

4. **Save and close the file**

### Step 3: Verify Security

✅ **The .env file is in .gitignore** - Your keys will NOT be committed to git  
✅ **No real keys in source code** - All keys are externalized  
✅ **Template files use placeholders** - Safe for version control  

### Required API Keys

| Service | Required | Purpose |
|---------|----------|---------|
| OpenAI API | **YES** | Core conversational AI features |
| Google Translate | Optional | Enhanced translation capabilities |
| Azure Translator | Optional | Alternative translation service |
| Hugging Face | Optional | Additional model access |

### Files That Depend on OpenAI Key

The following components require your OpenAI API key:

1. **Core Conversation System:**
   - `src/conversation/openai_chat_handler.py`
   - `src/conversation/openai_integration.py`
   - `src/conversation/ai_powered_conversation.py`

2. **Application Startup:**
   - `app.py` - Main Flask application
   - `run_newsbot.py` - Production startup script

3. **Configuration:**
   - `config/settings.py` - Configuration management

### Security Best Practices

1. **Never commit .env file to version control**
2. **Use different API keys for development/production**
3. **Monitor your OpenAI usage and billing**
4. **Rotate keys regularly**
5. **Use environment variables in production**

### Running Without OpenAI Key

If you don't have an OpenAI key:
- Basic classification and sentiment analysis will work
- Advanced conversational features will be disabled
- Translation features will use fallback services
- Some error messages will appear but won't break the system

### Troubleshooting

**Error: "OpenAI API key is required"**
- Check your .env file exists
- Verify the key is correctly formatted
- Ensure no extra spaces or quotes

**Error: "Invalid API key"**
- Verify the key is active on OpenAI platform
- Check for any billing issues
- Try regenerating the key

### Production Deployment

For production deployment:
```bash
# Set environment variables directly
export OPENAI_API_KEY="your-production-key"
export SECRET_KEY="your-super-secret-production-key"

# Or use secure key management
# AWS: aws ssm get-parameter --name "openai-key"
# Azure: az keyvault secret show --name "openai-key"
```

## 🛡️ Security Verification

Run this command to verify no sensitive data is in your repository:
```bash
# Check for any remaining keys
grep -r "sk-proj-" . --exclude-dir=.git --exclude="*.log"

# Should return no results
```

