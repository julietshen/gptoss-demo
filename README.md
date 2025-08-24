# gptoss-demo
a quick and dirty claude-coded open-source demonstration of AI-powered content moderation using locally-running Large Language Models (LLMs). This project showcases how to build an intelligent moderation system that can classify user-generated content, identify policy violations, and provide transparent reasoning for its decisions.

**🆕 NEW**: Now with **live AT Protocol integration** via Jetstream! Moderate real-time AT Protocol posts as they happen.


![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![Ollama](https://img.shields.io/badge/ollama-latest-green)
![AT Protocol](https://img.shields.io/badge/AT%20Protocol-Jetstream-blueviolet)
![License](https://img.shields.io/badge/license-MIT-purple)

## Overview

This demo application uses **GPT-OSS** running locally via **Ollama** to moderate user-generated content based on community guidelines. Now enhanced with **live AT Protocol integration** via Jetstream to process real-time AT Protocol posts. Developed and tested on a 32GB 2021 MacBook Pro.

### Core Capabilities
- **Real-time content classification** (Allowed / Flagged / Needs Human Review)
- **Live AT Protocol streaming** - Connect to AT Protocol's real-time data feed
- **Policy violation detection** with specific categories
- **Chain-of-thought reasoning** for transparent decision-making
- **Draft responses** for borderline cases requiring human review
- **Batch processing** capabilities for testing multiple posts
- **Live moderation dashboard** - Process actual social media posts as they happen

<img width="1498" height="733" alt="Screenshot 2025-08-07 at 5 44 10 PM" src="https://github.com/user-attachments/assets/689641cc-bf8b-424e-aab8-e7215547d2a0" />
<img width="1133" height="694" alt="Screenshot 2025-08-07 at 5 46 16 PM" src="https://github.com/user-attachments/assets/ea4bf6af-5639-4d74-9254-635f4decbe22" />


## Features

### Functionality
- **Local LLM Integration** - Runs entirely on your machine, no external API calls
- **Live AT Protocol Streaming** - Real-time connection to AT Protocol's data feed via Jetstream
- **Interactive UI** - Clean Streamlit interface for easy testing
- **Batch Demo Mode** - Process multiple posts automatically
- **Live Moderation Mode** - Process real-time social media posts as they happen
- **Detailed Reasoning** - Understand why content was flagged
- **Statistics Tracking** - Monitor moderation patterns across live and sample data
- **Debug Tools** - Built-in connection testing and diagnostics
- **Queue Management** - Buffer and process live posts efficiently

### Moderation Categories
- Hate Speech & Harassment
- Privacy Violations
- Spam & Commercial Content
- Misinformation
- Violence & Harmful Content
- And more based on [Bluesky Community Guidelines](https://bsky.social/about/support/community-guidelines)

## Prerequisites

- **Python 3.8+**
- **Ollama** installed and running
- **16GB+ RAM** recommended (32GB optimal for GPT-OSS:20b model)
- **macOS, Linux, or Windows** with WSL2

Tested on: 32GB 2021 MacBook Pro (M1 Max)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/julietshen/gptoss-demo.git
cd gptoss-demo
```

### 2. Install Python Dependencies
```bash
pip install streamlit requests datasets websocket-client zstandard
```

Or using uv (recommended):
```bash
uv pip install -r requirements.txt
```

Or using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Ollama
**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai)

### 4. Pull the GPT-OSS Model
```bash
ollama pull gpt-oss:20b
```

Note: This downloads approximately 13GB.

### 5. Verify Ollama is Running
```bash
ollama list  # Should show gpt-oss:20b
ollama run gpt-oss:20b "Hello"  # Test the model
```

## Usage

### Start the Application
```bash
streamlit run moderation_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Manual Moderation:**
   - Enter or paste content in the text box
   - Click "Analyze Post"
   - View the classification, reasoning, and recommendations

2. **Batch Demo:**
   - Click "Run Demo (10 posts)"
   - Watch as the system processes random sample posts
   - Review results in expandable sections

3. **🆕 Live AT Protocol Moderation:**
   - Click "Start Live Stream" to connect to AT Protocol's real-time data
   - Click "Moderate Live Posts" to process real-time posts
   - Watch live posts get moderated as they happen on the network
   - Monitor connection status and post queue in the sidebar

4. **Debug Mode:**
   - Click "Debug Connection" if you encounter issues
   - View detailed diagnostics about Ollama connection

### Testing Jetstream Connection

You can test the AT Protocol connection independently:

```bash
python test_jetstream.py
```

This will connect to Jetstream and display live posts as they come in, helping verify the integration works.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Jetstream      │
│  (Frontend)     │    │  WebSocket      │
└────────┬────────┘    └────────┬────────┘
         │                      │
         │              ┌───────▼────────┐
         │              │ AT Protocol    │
         │              │ Live Stream    ││
         │              └───────┬────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────┐
│         Moderation Engine               │
│  (Live + Sample Data Processing)       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────┐
│  Ollama API     │
│  (localhost)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPT-OSS:20b    │
│  (Local LLM)    │
└─────────────────┘
```

## Project Structure

```
trust-safety-demo/
├── moderation_app.py     # Main Streamlit application
├── jetstream_client.py   # AT Protocol/Jetstream WebSocket client
├── test_jetstream.py     # Standalone Jetstream connection test
├── README.md            # Documentation
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore file
```

## How It Works

### Traditional Mode
1. **Content Input**: User provides text to moderate or runs batch demo
2. **Prompt Engineering**: System constructs a detailed prompt with:
   - Bluesky Community Guidelines context
   - Structured output format requirements
   - Chain-of-thought reasoning instructions
3. **LLM Processing**: GPT-OSS analyzes the content
4. **Response Parsing**: System extracts:
   - Classification decision
   - Violation category (if any)
   - Reasoning explanation
   - Draft user response (for edge cases)
5. **UI Display**: Results shown with appropriate visual indicators

### 🆕 Live AT Protocol Mode
1. **Stream Connection**: WebSocket connects to Jetstream (real-time AT Protocol feed)
2. **Data Filtering**: Incoming messages filtered for `app.bsky.feed.post` (text posts)
3. **Queue Management**: Posts buffered in a thread-safe queue for processing
4. **Real-time Moderation**: Live posts processed through the same GPT-OSS pipeline
5. **Live Dashboard**: Real-time stats and moderation results displayed as posts happen

## Configuration

### Changing the Model
Edit `MODEL_NAME` in `moderation_app.py`:
```python
MODEL_NAME = "gpt-oss:20b"  # Or any Ollama model
```

### Adjusting Temperature
Modify the generation parameters:
```python
payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.3,  # Lower = more consistent
    }
}
```

### Custom Sample Posts
Edit the `SAMPLE_POSTS` list to test specific scenarios

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Check if Ollama is running
ollama list

# Start Ollama manually
ollama serve
```

### "Model not found (404)"
```bash
# Install the model
ollama pull gpt-oss:20b

# Verify installation
ollama list
```

### Slow Performance
- Increase timeout in `moderation_app.py`
- Check available RAM
- Ensure no other heavy processes are running

### Debug Connection Issues
Use the built-in debug button in the UI or run:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Test",
  "stream": false
}'
```

## Performance Considerations

Testing environment: 32GB 2021 MacBook Pro
- **GPT-OSS:20b**: ~13GB model size, 30-60 seconds per moderation
- **RAM Usage**: 16-20GB during operation
- **CPU/GPU**: Utilizes Apple Silicon acceleration when available
- **Live Streaming**: Minimal additional overhead, WebSocket connection is lightweight
- **Queue Processing**: Efficiently handles bursts of real-time posts


## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Streamlit Documentation](https://docs.streamlit.io)
- [GPT-OSS Model Info](https://huggingface.co/openai/gpt-oss-20b)
- [Bluesky Community Guidelines](https://bsky.social/about/support/community-guidelines)
- [Bluesky dataset of 8m posts](https://huggingface.co/datasets/withalim/bluesky-posts)
- [Jetstream AT Protocol Documentation](https://github.com/bluesky-social/jetstream)
- [AT Protocol Specification](https://atproto.com)

## Disclaimer

This is a **demonstration system** for educational purposes. Production content moderation systems require:
- Comprehensive testing and validation
- Human review workflows
- Appeals processes
- Compliance with regional regulations
- Regular model updates and monitoring
- Bias testing and mitigation strategies

## Acknowledgments

- OpenAI for GPT-OSS model
- Ollama team for local LLM infrastructure
- Streamlit for the UI framework
- Bluesky team for community guidelines inspiration and 
- Jaz for the Jetstream AT Protocol infrastructure
- Alim Maasoglu for curating and collecting the Bluesky data
- AT Protocol team for the decentralized social networking specification
