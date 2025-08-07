"""
Trust & Safety Agentic Workflow Demo
A Streamlit application for content moderation using GPT-OSS via Ollama
Author: AI Safety Demo
"""

import streamlit as st
import requests
import json
import random
import logging
from datetime import datetime
from typing import Dict, Optional, List
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('moderation_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"  # ONLY use gpt-oss:20b
DATASET_NAME = "withalim/bluesky-posts"

# Sample posts for demo (fallback if dataset loading fails)
SAMPLE_POSTS = [
    "Just had the best coffee at my local cafe! ☕ #morningvibes",
    "I hate all people who disagree with me. They should be eliminated.",
    "Check out my new blog post about machine learning fundamentals!",
    "CLICK HERE TO WIN $1000!!! LIMITED TIME OFFER!!!",
    "Anyone know a good restaurant in downtown Seattle?",
    "Sharing personal info: John Doe lives at 123 Main St, SSN: 123-45-6789",
    "The earth is flat and vaccines cause autism. Wake up sheeple!",
    "Beautiful sunset today 🌅 Nature is amazing!",
    "Kill yourself if you don't agree with my opinion",
    "Looking for collaborators on my open source project",
]

def build_prompt(post_text: str) -> str:
    """
    Build the moderation prompt for GPT-OSS.
    
    Args:
        post_text: The user-generated content to moderate
    
    Returns:
        Formatted prompt string for the model
    """
    prompt = f"""You are a trust & safety moderation assistant for a social media platform. Your job is to review a user-generated post and decide:

1. Whether it violates Bluesky's Community Guidelines.
2. If it violates, what category of policy it violates (e.g., Hate, Harassment, Privacy Violation, Spam, Misinformation).
3. Explain your reasoning in a few sentences.
4. If it's unclear, provide a draft response to the user. If it's clearly okay or clearly violating, say 'N/A'.

Return your result in this exact format:
Classification: [Allowed / Flagged / Needs Human Review]
Violation Type: [Type or N/A]
Rationale: [Short explanation]
Draft Response: [Message to user or N/A]

Post: \"\"\"{post_text}\"\"\"
"""
    return prompt

def parse_model_response(response_text: str) -> Dict[str, str]:
    """
    Parse the model's response into structured format.
    
    Args:
        response_text: Raw text response from the model
    
    Returns:
        Dictionary with parsed classification, violation type, rationale, and draft response
    """
    result = {
        "classification": "Unknown",
        "violation_type": "N/A",
        "rationale": "Unable to parse response",
        "draft_response": "N/A"
    }
    
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith("Classification:"):
                result["classification"] = line.replace("Classification:", "").strip()
            elif line.startswith("Violation Type:"):
                result["violation_type"] = line.replace("Violation Type:", "").strip()
            elif line.startswith("Rationale:"):
                result["rationale"] = line.replace("Rationale:", "").strip()
            elif line.startswith("Draft Response:"):
                result["draft_response"] = line.replace("Draft Response:", "").strip()
    except Exception as e:
        logger.error(f"Error parsing model response: {e}")
    
    return result

def check_ollama_status() -> tuple[bool, str, list]:
    """
    Check if Ollama is running and list available models.
    
    Returns:
        Tuple of (is_running, status_message, available_models)
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return True, "Ollama is running", model_names
        else:
            return False, f"Ollama returned status {response.status_code}", []
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama (not running?)", []
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}", []

def moderate_post(text: str) -> Dict[str, str]:
    """
    Send post to GPT-OSS for moderation via Ollama API.
    
    Args:
        text: The post text to moderate
    
    Returns:
        Dictionary with moderation results
    """
    if not text or not text.strip():
        return {
            "classification": "Error",
            "violation_type": "N/A",
            "rationale": "Empty post provided",
            "draft_response": "N/A"
        }
    
    prompt = build_prompt(text)
    
    try:
        logger.info(f"Moderating post: {text[:50]}...")
        
        # First check if Ollama is running and model exists
        is_running, status_msg, models = check_ollama_status()
        if not is_running:
            logger.error(f"Ollama check failed: {status_msg}")
            return {
                "classification": "Error",
                "violation_type": "N/A",
                "rationale": f"Ollama not available: {status_msg}. Please start Ollama with 'ollama serve'",
                "draft_response": "N/A"
            }
        
        # Log available models for debugging
        logger.info(f"Available models: {models}")
        
        # Prepare request payload - ensure exact format
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        # Log the request for debugging
        logger.info(f"Sending request to {OLLAMA_API_URL} with model: {MODEL_NAME}")
        
        # Make API call to Ollama
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=60  # Increased timeout for larger model
        )
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            model_output = response_data.get("response", "")
            logger.info("Successfully received moderation response")
            return parse_model_response(model_output)
        elif response.status_code == 404:
            # Log more details about the 404
            logger.error(f"404 Error. Response: {response.text}")
            logger.error(f"Request URL: {OLLAMA_API_URL}")
            logger.error(f"Model name: {MODEL_NAME}")
            logger.error(f"Available models: {models}")
            
            # Check if model name format is issue
            if "gpt-oss" in MODEL_NAME and "gpt-oss:20b" not in models:
                # Try alternative formats
                alt_names = ["gpt-oss:20b", "gpt-oss", "gpt-oss:latest"]
                for alt_name in alt_names:
                    if alt_name in models:
                        logger.info(f"Found model under name: {alt_name}")
                        return {
                            "classification": "Error",
                            "violation_type": "N/A",
                            "rationale": f"Model name mismatch. Try using '{alt_name}' instead of '{MODEL_NAME}'",
                            "draft_response": "N/A"
                        }
            
            return {
                "classification": "Error",
                "violation_type": "N/A",
                "rationale": f"Model endpoint not found (404). Check model name format. Available: {', '.join(models[:3]) if models else 'none'}",
                "draft_response": "N/A"
            }
        else:
            error_msg = response.text if response.text else f"Status {response.status_code}"
            logger.error(f"Ollama API error: {error_msg}")
            return {
                "classification": "Error",
                "violation_type": "N/A",
                "rationale": f"API Error: {error_msg}",
                "draft_response": "N/A"
            }
            
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Make sure Ollama is running locally.")
        return {
            "classification": "Error",
            "violation_type": "N/A",
            "rationale": "Cannot connect to Ollama. Please ensure Ollama is running at localhost:11434",
            "draft_response": "N/A"
        }
    except requests.exceptions.Timeout:
        logger.error("Request to Ollama timed out")
        return {
            "classification": "Error",
            "violation_type": "N/A",
            "rationale": "Request timed out. The model may be loading or overloaded.",
            "draft_response": "N/A"
        }
    except Exception as e:
        logger.error(f"Unexpected error during moderation: {e}")
        return {
            "classification": "Error",
            "violation_type": "N/A",
            "rationale": f"Unexpected error: {str(e)}",
            "draft_response": "N/A"
        }

def load_sample_posts() -> List[str]:
    """
    Load sample posts from the dataset or use fallback samples.
    
    Returns:
        List of sample post texts
    """
    try:
        from datasets import load_dataset
        logger.info("Loading posts from Hugging Face dataset...")
        
        # Attempt to load the dataset
        dataset = load_dataset(DATASET_NAME, split="train[:100]")
        posts = [item.get("text", item.get("content", "")) for item in dataset]
        posts = [p for p in posts if p and len(p.strip()) > 0][:100]
        
        if posts:
            logger.info(f"Successfully loaded {len(posts)} posts from dataset")
            return posts
        else:
            logger.warning("Dataset loaded but no valid posts found, using fallback")
            return SAMPLE_POSTS
            
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}. Using fallback sample posts.")
        return SAMPLE_POSTS

def display_moderation_result(result: Dict[str, str]):
    """
    Display moderation results in the Streamlit UI with appropriate styling.
    
    Args:
        result: Dictionary containing moderation results
    """
    # Color coding for classification
    classification = result.get("classification", "Unknown")
    
    if classification == "Allowed":
        st.success(f"**Classification:** {classification} ✅")
    elif classification == "Flagged":
        st.error(f"**Classification:** {classification} 🚫")
    elif classification == "Needs Human Review":
        st.warning(f"**Classification:** {classification} ⚠️")
    else:
        st.info(f"**Classification:** {classification}")
    
    # Display other fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Violation Type:**")
        st.write(result.get("violation_type", "N/A"))
    
    with col2:
        st.markdown("**Rationale:**")
        st.write(result.get("rationale", "No rationale provided"))
    
    # Display draft response if applicable
    if result.get("draft_response", "N/A") != "N/A":
        st.markdown("**Draft Response to User:**")
        st.info(result.get("draft_response"))

def test_ollama_connection():
    """
    Test function to debug Ollama connection issues.
    """
    st.header("🔍 Debugging Ollama Connection")
    
    # Test 1: Check if Ollama is running
    st.write("**Test 1: Ollama Service**")
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        st.success(f"✅ Ollama is responding (status: {response.status_code})")
    except:
        st.error("❌ Cannot connect to Ollama")
    
    # Test 2: List models
    st.write("**Test 2: Available Models**")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                st.code(model.get("name", "unknown"))
    except Exception as e:
        st.error(f"Error listing models: {e}")
    
    # Test 3: Try a simple generation
    st.write("**Test 3: Test Generation**")
    test_payload = {
        "model": MODEL_NAME,
        "prompt": "Say 'test successful' and nothing else.",
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=test_payload,
            timeout=30
        )
        st.write(f"Response status: {response.status_code}")
        if response.status_code == 200:
            st.success("✅ Generation endpoint works!")
            st.code(response.json().get("response", "No response"))
        else:
            st.error(f"❌ Error: {response.status_code}")
            st.code(response.text)
    except Exception as e:
        st.error(f"Error testing generation: {e}")

def main():
    """
    Main Streamlit application function.
    """
    # Page configuration
    st.set_page_config(
        page_title="Trust & Safety Moderation Demo",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Header
    st.title("🛡️ Trust & Safety Agentic Workflow Demo")
    st.markdown("*Content moderation using GPT-OSS:20b via Ollama*")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **How to use:**
        1. Enter text in the input box OR
        2. Click 'Run Demo' to moderate random posts
        3. View the AI's moderation decision
        
        **Classification Types:**
        - ✅ **Allowed**: Content is safe
        - 🚫 **Flagged**: Clear violation
        - ⚠️ **Needs Review**: Borderline case
        
        **Prerequisites:**
        - Ollama running locally
        - Model: `gpt-oss:20b`
        
        **Based on Bluesky Community Guidelines**
        """)
        
        st.divider()
        
        # System status
        st.header("🔧 System Status")
        is_running, status_msg, available_models = check_ollama_status()
        
        if is_running:
            st.success(f"✅ {status_msg}")
            if available_models:
                st.info(f"**Available models:** {', '.join(available_models[:3])}")
                if MODEL_NAME not in available_models:
                    st.warning(f"⚠️ Model '{MODEL_NAME}' not installed")
                    st.code(f"ollama pull {MODEL_NAME}", language="bash")
            else:
                st.warning("No models installed")
        else:
            st.error(f"❌ {status_msg}")
            st.code("ollama serve", language="bash")
            st.info("Start Ollama first!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Content Input")
        
        # Text input
        user_input = st.text_area(
            "Enter content to moderate:",
            height=150,
            placeholder="Type or paste user-generated content here..."
        )
        
        # Buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            analyze_button = st.button("🔍 Analyze Post", type="primary", use_container_width=True)
        
        with button_col2:
            demo_button = st.button("🎲 Run Demo (10 posts)", use_container_width=True)
        
        with button_col3:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        # Add debug button
        debug_button = st.button("🔧 Debug Connection", use_container_width=True)
    
    with col2:
        st.header("📊 Quick Stats")
        if "total_moderated" not in st.session_state:
            st.session_state.total_moderated = 0
            st.session_state.allowed_count = 0
            st.session_state.flagged_count = 0
            st.session_state.review_count = 0
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Moderated", st.session_state.total_moderated)
            st.metric("Allowed", st.session_state.allowed_count)
        with metric_col2:
            st.metric("Flagged", st.session_state.flagged_count)
            st.metric("Needs Review", st.session_state.review_count)
    
    # Handle button actions
    if clear_button:
        st.rerun()
    
    if debug_button:
        test_ollama_connection()
    
    if analyze_button:
        if user_input.strip():
            with st.spinner("🤖 Analyzing content..."):
                result = moderate_post(user_input)
                
                # Update stats
                st.session_state.total_moderated += 1
                if result["classification"] == "Allowed":
                    st.session_state.allowed_count += 1
                elif result["classification"] == "Flagged":
                    st.session_state.flagged_count += 1
                elif result["classification"] == "Needs Human Review":
                    st.session_state.review_count += 1
                
                st.divider()
                st.header("🎯 Moderation Result")
                display_moderation_result(result)
        else:
            st.error("Please enter some content to moderate.")
    
    if demo_button:
        st.divider()
        st.header("🎲 Demo: Moderating Random Posts")
        
        # Load sample posts
        sample_posts = load_sample_posts()
        
        # Select 10 random posts
        selected_posts = random.sample(sample_posts, min(10, len(sample_posts)))
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each post
        for i, post in enumerate(selected_posts):
            status_text.text(f"Processing post {i+1} of {len(selected_posts)}...")
            progress_bar.progress((i + 1) / len(selected_posts))
            
            with st.expander(f"Post {i+1}: {post[:50]}...", expanded=(i==0)):
                st.markdown("**Original Post:**")
                st.write(post)
                
                with st.spinner("Moderating..."):
                    result = moderate_post(post)
                    
                    # Update stats
                    st.session_state.total_moderated += 1
                    if result["classification"] == "Allowed":
                        st.session_state.allowed_count += 1
                    elif result["classification"] == "Flagged":
                        st.session_state.flagged_count += 1
                    elif result["classification"] == "Needs Human Review":
                        st.session_state.review_count += 1
                    
                    display_moderation_result(result)
            
            # Small delay to prevent overwhelming the API
            time.sleep(0.5)
        
        status_text.text("✅ Demo complete!")
        st.balloons()
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    *Built with Streamlit, Ollama, and GPT-OSS | Based on Bluesky Community Guidelines*
    
    **Note:** This is a demonstration system. Production moderation systems require additional safeguards, 
    human review processes, and compliance with applicable regulations.
    """)

if __name__ == "__main__":
    main()
