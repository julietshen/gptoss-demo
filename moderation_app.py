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
import queue
import threading
from jetstream_client import JetstreamClient, create_post_stream_client, filter_posts
from at_labeler import ATLabelerClient, AgenticModerator, create_demo_labeler
from label_config import LabelConfig, AgenticPolicy
from feedback_system import FeedbackSystem, FeedbackType, FeedbackAnalyzer
from guidelines_manager import GuidelinesManager, GuidelineSet

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

# Sample posts removed per user request - using only HuggingFace dataset or live Jetstream data
SAMPLE_POSTS = []

def build_prompt(post_text: str, guidelines_manager: GuidelinesManager = None, custom_guidelines: str = None) -> str:
    """
    Build the moderation prompt for GPT-OSS.
    
    Args:
        post_text: The user-generated content to moderate
        guidelines_manager: Optional guidelines manager for custom guidelines
    
    Returns:
        Formatted prompt string for the model
    """
    # Use custom guidelines if available, otherwise use default
    if guidelines_manager:
        guidelines_section = guidelines_manager.generate_prompt_section()
        draft_response = guidelines_manager.get_draft_response_template()
    else:
        # Fallback to default guidelines
        guidelines_section = """**Moderation Guidelines: Anti-Misogyny & Anti-Transphobia**

**Key Categories to Detect:**

**Misogyny:**
- Explicit: Direct slurs (b*tch, c*nt, wh*re), statements of female inferiority, calls for violence against women
- Subtle: Dismissive language ("Karen", "emotional", "hysterical"), capability generalizations, victim-blaming
- Harassment: Targeted gendered attacks, coordinated harassment, threats of sexual violence

**Transphobia:**
- Explicit: Deliberate misgendering, slurs (tr*nny), denying trans existence, calls for violence
- Subtle: "Biological sex" exclusion arguments, "real women/men" language, bathroom/sports fearmongering
- Deadnaming: Using former names/identities of trans people
- Medicalization: Calling being trans a "mental illness", unsolicited medical advice, detransition promotion

**Decision Principles:**
- Focus on impact over intent
- Consider intersectionality (racism + misogyny, etc.)
- Distinguish education from perpetuation
- Account for reclaimed language by affected communities"""
        draft_response = "pls dont post this tyvm"
    
    # Add custom guidelines if provided
    if custom_guidelines and custom_guidelines.strip():
        guidelines_section += f"""

**Additional Custom Guidelines:**
{custom_guidelines.strip()}"""
    
    prompt = f"""You are a specialized trust & safety moderation assistant. ANALYZE THE FOLLOWING POST NOW and provide your moderation decision:

TASK: Review this user-generated post and determine if it violates community guidelines.

INSTRUCTIONS:
1. Analyze the content for policy violations based on the guidelines below
2. Identify the specific type of violation if present
3. Provide your reasoning for the decision
4. If the post is not in English, first translate it to English and include the translation in your rationale

{guidelines_section}

RESPOND IMMEDIATELY in this exact format (do not ask for more information):
Classification: [Allowed / Flagged / Needs Human Review]
Violation Type: [Specific violation type or N/A]
Rationale: [If non-English: "Translation: [English translation]" then your detailed explanation of decision]
Draft Response: [If content violates policy, respond with "{draft_response}", otherwise N/A]

POST TO ANALYZE: \"\"\"{post_text}\"\"\"

PROVIDE YOUR MODERATION DECISION NOW:
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
        
        # Log the raw response for debugging if parsing fails
        if result["classification"] == "Unknown":
            logger.warning(f"Failed to parse model response. Raw response: {response_text[:200]}...")
            # If we can't parse the structured format, use the raw response as rationale
            result["rationale"] = f"Raw response: {response_text[:500]}..."
            
    except Exception as e:
        logger.error(f"Error parsing model response: {e}")
        result["rationale"] = f"Parse error: {str(e)} | Raw: {response_text[:200]}..."
    
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

def moderate_post(text: str, post_uri: str = None) -> Dict[str, str]:
    """
    Send post to GPT-OSS for moderation via Ollama API.
    
    Args:
        text: The post text to moderate
        post_uri: Optional AT Protocol URI for the post
    
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
    
    # Get policy information from session state (safely)
    try:
        if 'st' in globals() and hasattr(st, 'session_state'):
            policy_type = st.session_state.get("active_policy_type", "default")
            custom_policy = st.session_state.get("custom_policy_text", "")
            custom_guidelines = st.session_state.get("custom_guidelines_text", "")
        else:
            policy_type = "default"
            custom_policy = ""
            custom_guidelines = ""
    except Exception:
        # Fallback if session state access fails
        policy_type = "default"
        custom_policy = ""
        custom_guidelines = ""
    
    # Build prompt based on active policy type
    if policy_type == "custom_policy" and custom_policy.strip():
        # Use completely custom policy
        prompt = f"""You are a specialized trust & safety moderation assistant. ANALYZE THE FOLLOWING POST NOW and provide your moderation decision:

TASK: Review this user-generated post and determine if it violates community guidelines.

INSTRUCTIONS:
1. Analyze the content for policy violations based on the guidelines below
2. Identify the specific type of violation if present
3. Provide your reasoning for the decision
4. If the post is not in English, first translate it to English and include the translation in your rationale

{custom_policy}

RESPOND IMMEDIATELY in this exact format (do not ask for more information):
Classification: [Allowed / Flagged / Needs Human Review]
Violation Type: [Specific violation type or N/A]
Rationale: [If non-English: "Translation: [English translation]" then your detailed explanation of decision]
Draft Response: [If content violates policy, respond with "pls dont post this tyvm", otherwise N/A]

POST TO ANALYZE: \"\"\"{text}\"\"\"

PROVIDE YOUR MODERATION DECISION NOW:
"""
        logger.info("Using custom policy")
    else:
        # Use default policy with optional additional guidelines
        prompt = build_prompt(text, None, custom_guidelines if policy_type == "add_guidelines" else "")
        if policy_type == "add_guidelines" and custom_guidelines.strip():
            logger.info("Using default prompt with custom guidelines")
        else:
            logger.info("Using default prompt")
    
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
            logger.info(f"Model response preview: {model_output[:200]}...")
            parsed_result = parse_model_response(model_output)
            logger.info(f"Parsed classification: {parsed_result.get('classification')}")
            return parsed_result
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
            logger.warning("Dataset loaded but no valid posts found")
            return []
            
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}. No fallback posts available.")
        return []

def initialize_jetstream_state():
    """Initialize Jetstream session state if not already done."""
    if "jetstream_queue" not in st.session_state:
        st.session_state.jetstream_queue = queue.Queue(maxsize=1000)
        st.session_state.jetstream_client = None
        st.session_state.jetstream_thread = None
        st.session_state.jetstream_connected = False
        st.session_state.live_posts = []
        st.session_state.total_live_posts = 0

def start_jetstream_monitoring():
    """
    Start monitoring live AT Protocol data from Jetstream.
    """
    initialize_jetstream_state()
    
    # Create a thread-safe queue reference
    message_queue = st.session_state.jetstream_queue
    
    def on_post_received(post_data):
        try:
            if not message_queue.full():
                message_queue.put(post_data)
                # Update counter safely
                if "total_live_posts" in st.session_state:
                    st.session_state.total_live_posts += 1
        except Exception as e:
            logger.error(f"Error queueing post: {e}")
    
    def on_connect():
        if "jetstream_connected" in st.session_state:
            st.session_state.jetstream_connected = True
        logger.info("Connected to Jetstream")
    
    def on_disconnect():
        if "jetstream_connected" in st.session_state:
            st.session_state.jetstream_connected = False
        logger.info("Disconnected from Jetstream")
    
    def on_error(error):
        logger.error(f"Jetstream error: {error}")
        if "jetstream_connected" in st.session_state:
            st.session_state.jetstream_connected = False
    
    if not st.session_state.jetstream_client:
        client = create_post_stream_client(on_post_received)
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_error = on_error
        
        st.session_state.jetstream_client = client
        st.session_state.jetstream_thread = client.start()
        # Set connected status immediately (connection will be established shortly)
        st.session_state.jetstream_connected = True
        logger.info("Started Jetstream client")

def stop_jetstream_monitoring():
    """
    Stop monitoring Jetstream.
    """
    if st.session_state.get("jetstream_client"):
        st.session_state.jetstream_client.disconnect()
        st.session_state.jetstream_client = None
        st.session_state.jetstream_connected = False
        st.session_state.jetstream_thread = None
        logger.info("Stopped Jetstream client")

def get_live_posts(max_posts: int = 10) -> List[Dict]:
    """
    Get live posts from Jetstream queue.
    
    Args:
        max_posts: Maximum number of posts to retrieve
        
    Returns:
        List of post dictionaries
    """
    posts = []
    
    if "jetstream_queue" not in st.session_state:
        return posts
    
    try:
        while len(posts) < max_posts and not st.session_state.jetstream_queue.empty():
            post_data = st.session_state.jetstream_queue.get_nowait()
            if post_data and post_data.get("text", "").strip():
                posts.append(post_data)
    except queue.Empty:
        pass
    except Exception as e:
        logger.error(f"Error retrieving live posts: {e}")
    
    return posts

def process_with_agentic_labeling(moderation_result: Dict[str, str], post_uri: str = None) -> Dict[str, any]:
    """
    Process moderation result with agentic labeling if enabled.
    
    Args:
        moderation_result: Result from moderate_post()
        post_uri: AT Protocol URI for the post
        
    Returns:
        Enhanced result with labeling information
    """
    enhanced_result = moderation_result.copy()
    enhanced_result["agentic_result"] = None
    
    # Only process if agentic labeling is enabled and we have a valid result
    if (st.session_state.get("agentic_enabled", False) and 
        "agentic_moderator" in st.session_state and
        moderation_result.get("classification") != "Error"):
        
        try:
            # Create a mock URI if none provided
            if not post_uri:
                post_uri = f"at://did:plc:demo/app.bsky.feed.post/{int(time.time())}"
            
            # Process with agentic moderator
            agentic_result = st.session_state.agentic_moderator.process_moderation_result(
                post_uri, moderation_result
            )
            enhanced_result["agentic_result"] = agentic_result
            
        except Exception as e:
            logger.error(f"Error in agentic processing: {e}")
    
    return enhanced_result

def show_feedback_form(result: Dict[str, str], post_text: str, post_uri: str = None):
    """
    Show feedback form for a moderation result.
    
    Args:
        result: Moderation result dictionary
        post_text: Original post text
        post_uri: Optional post URI
    """
    if not post_uri:
        post_uri = f"demo://post/{int(time.time())}"
    
    st.divider()
    st.subheader("📝 Provide Feedback")
    
    with st.form(f"feedback_form_{hash(post_text)}", clear_on_submit=True):
        st.write("**Help improve our moderation by providing feedback:**")
        
        # Feedback type selection
        feedback_type = st.selectbox(
            "What type of feedback is this?",
            options=[
                ("false_positive", "False Positive - AI incorrectly flagged safe content"),
                ("false_negative", "False Negative - AI missed harmful content"),
                ("wrong_label", "Wrong Label - AI applied incorrect label type"),
                ("severity_wrong", "Wrong Severity - AI got the severity level wrong")
            ],
            format_func=lambda x: x[1]
        )
        
        # Correct classification
        correct_classification = st.selectbox(
            "What should the correct classification be?",
            ["Allowed", "Flagged", "Needs Human Review"]
        )
        
        # Correct labels
        available_labels = list(LabelConfig.LABELS.keys())
        correct_labels = st.multiselect(
            "What labels should be applied? (if any)",
            available_labels,
            help="Select all labels that should apply to this content"
        )
        
        # User explanation
        user_explanation = st.text_area(
            "Please explain your feedback:",
            placeholder="Describe why you think the AI's decision was incorrect and what the correct decision should be...",
            height=100
        )
        
        # Confidence rating
        confidence = st.slider(
            "How confident are you in this feedback?",
            min_value=1,
            max_value=5,
            value=4,
            help="1 = Not very confident, 5 = Very confident"
        )
        
        # Submit button
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            if user_explanation.strip():
                # Submit feedback
                feedback_id = st.session_state.feedback_system.submit_feedback(
                    post_text=post_text,
                    post_uri=post_uri,
                    original_classification=result.get("classification", "Unknown"),
                    original_labels=result.get("agentic_result", {}).get("labels_assigned", []),
                    feedback_type=FeedbackType(feedback_type[0]),
                    correct_classification=correct_classification,
                    correct_labels=correct_labels,
                    user_explanation=user_explanation,
                    confidence=confidence
                )
                
                st.success(f"✅ Feedback submitted successfully! ID: {feedback_id}")
                st.info("Thank you for helping improve our moderation system!")
                
                # Show immediate analysis if this creates a pattern
                analyzer = st.session_state.feedback_analyzer
                suggestions = analyzer.get_improvement_suggestions()
                if suggestions:
                    with st.expander("💡 Improvement Suggestions Based on Feedback"):
                        for suggestion in suggestions:
                            st.write(f"• {suggestion}")
                            
            else:
                st.error("Please provide an explanation for your feedback.")

def display_moderation_result(result: Dict[str, str], post_text: str = "", post_uri: str = None):
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
    
    # Display agentic labeling results if available
    if result.get("agentic_result"):
        st.divider()
        agentic_result = result["agentic_result"]
        
        st.markdown("**🤖 Agentic Labeling Result:**")
        
        col_x, col_y = st.columns(2)
        with col_x:
            st.metric("AI Confidence", f"{agentic_result.get('confidence', 0):.0%}")
            st.write(f"**Action:** {agentic_result.get('action_taken', 'none')}")
        
        with col_y:
            if agentic_result.get("labels_assigned"):
                st.success(f"**Labels Assigned:** {', '.join(agentic_result['labels_assigned'])}")
            else:
                st.info("**Labels:** None assigned")
        
        if agentic_result.get("requires_human_review"):
            st.warning("⚠️ **Flagged for Human Review**")
        
        # Show label details
        if agentic_result.get("labels_assigned"):
            with st.expander("Label Details"):
                for label_id in agentic_result["labels_assigned"]:
                    label = LabelConfig.get_label(label_id)
                    if label:
                        st.markdown(f"**{label.name}** ({label.severity.value})")
                        st.write(f"_{label.description}_")
    
    # Add feedback button
    if post_text and "feedback_system" in st.session_state:
        col_feedback1, col_feedback2, col_feedback3 = st.columns([1, 1, 2])
        
        with col_feedback1:
            if st.button("📝 Provide Feedback", key=f"feedback_btn_{hash(post_text)}", help="Report incorrect moderation decisions"):
                st.session_state[f"show_feedback_{hash(post_text)}"] = True
        
        with col_feedback2:
            if st.button("✅ Decision Correct", key=f"correct_btn_{hash(post_text)}", help="Mark this decision as correct"):
                st.success("Thank you for confirming this decision is correct!")
        
        # Show feedback form if requested
        if st.session_state.get(f"show_feedback_{hash(post_text)}", False):
            show_feedback_form(result, post_text, post_uri)
            if st.button("❌ Cancel Feedback", key=f"cancel_feedback_{hash(post_text)}"):
                st.session_state[f"show_feedback_{hash(post_text)}"] = False
                st.rerun()

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
        page_title="ATProto NYC Hack Day Demo",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Header
    st.title("🛡️ ATProto NYC Hack Day Demo")
    st.markdown("*Generalizable classification using GPT-OSS with live AT Protocol data*")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **How to use:**
        1. Enter text in the input box OR
        2. Click 'Run Demo' to moderate dataset posts OR
        3. Click 'Start Jetstream' to connect to AT Protocol
        4. Click 'Moderate Live Posts' to process real-time data
        5. View the AI's moderation decision
        6. Use 'Provide Feedback' to report incorrect decisions
        
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
    # Add guidelines management tab
    tab1, tab2 = st.tabs(["🛡️ Moderation", "⚙️ Policy Settings"])
    
    with tab2:
        st.header("⚙️ Policy Settings")
        st.markdown("Choose or create your moderation policy")
        
        # Policy note
        st.info("📢 **Note:** All policies allow promotion of other channels, sites, and self-promotion. Only malicious links and scams are flagged.")
        
        # Initialize policy state
        if "active_policy_type" not in st.session_state:
            st.session_state.active_policy_type = "default"
        if "custom_policies" not in st.session_state:
            st.session_state.custom_policies = {}
        if "custom_guidelines_text" not in st.session_state:
            st.session_state.custom_guidelines_text = ""
        
        # Policy Type Selection
        st.subheader("📋 Policy Type")
        
        policy_type = st.selectbox(
            "Choose your moderation approach:",
            options=["default", "custom_policy", "add_guidelines"],
            format_func=lambda x: {
                "default": "Default - Anti-Misogyny & Anti-Transphobia",
                "custom_policy": "Custom Policy - Replace with your own guidelines",
                "add_guidelines": "Enhanced Default - Add custom guidelines to default policy"
            }[x],
            index=["default", "custom_policy", "add_guidelines"].index(st.session_state.active_policy_type) if st.session_state.active_policy_type in ["default", "custom_policy", "add_guidelines"] else 0
        )
        
        if policy_type == "custom_policy":
            # Custom Policy Creation
            st.subheader("📝 Create Custom Policy")
            st.markdown("Paste your complete moderation policy. This will replace the default guidelines entirely.")
            
            # Policy templates
            with st.expander("📋 Policy Templates (Click to use)"):
                if st.button("🎮 Livestream Chat Abuse"):
                    template = """**Livestream Chat Moderation Policy**

Flag messages containing:

**Spam & Disruption:**
- Repeated identical or similar messages
- Excessive use of emotes or symbols
- Attempts to disrupt chat flow or derail conversation

**Harassment & Toxicity:**
- Personal attacks against streamers or viewers
- Targeted harassment or bullying
- Doxxing attempts or sharing personal information

**Inappropriate Content:**
- Sexual harassment or inappropriate advances
- Hate speech targeting individuals or groups
- Threats or incitement to violence

**Platform Violations:**
- Scam links or suspicious URLs
- Impersonation of streamers or moderators
- Malicious or harmful content

**Backseat Gaming & Spoilers:**
- Unwanted gameplay advice when prohibited
- Spoilers for games, movies, or shows
- Excessive criticism of gameplay choices"""
                    st.session_state.custom_policy_text = template
                    st.rerun()
                
                if st.button("💼 Professional Community"):
                    template = """**Professional Community Moderation Policy**

Flag content containing:

**Unprofessional Conduct:**
- Personal attacks or hostile behavior
- Inappropriate language or content
- Off-topic discussions unrelated to professional matters

**Spam & Repetitive Content:**
- Repetitive identical or similar posts
- Irrelevant posts unrelated to community purpose
- Excessive posting that disrupts normal discussion

**Misinformation & Harmful Advice:**
- False or misleading professional advice
- Unsubstantiated claims about companies or individuals
- Sharing of confidential or proprietary information

**Discrimination & Harassment:**
- Discriminatory language or behavior
- Harassment based on protected characteristics
- Creating hostile work environment discussions"""
                    st.session_state.custom_policy_text = template
                    st.rerun()
            
            # Custom policy text area
            if "custom_policy_text" not in st.session_state:
                st.session_state.custom_policy_text = ""
            
            custom_policy = st.text_area(
                "Paste or write your custom moderation policy:",
                value=st.session_state.custom_policy_text,
                height=300,
                placeholder="Paste your complete moderation guidelines here...\n\nExample:\n**My Custom Policy**\n\nFlag content containing:\n- Specific rule 1\n- Specific rule 2\n- etc.",
                help="This will completely replace the default guidelines. Make sure to include all the rules you want enforced."
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save Custom Policy", type="primary"):
                    st.session_state.custom_policy_text = custom_policy
                    st.session_state.active_policy_type = "custom_policy"
                    st.success("✅ Custom policy activated!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Back to Default"):
                    st.session_state.active_policy_type = "default"
                    st.success("✅ Switched back to default policy!")
                    st.rerun()
        
        elif policy_type == "add_guidelines":
            # Add to default guidelines
            st.subheader("📝 Add Custom Guidelines")
            st.markdown("Add specific guidelines to supplement the default policy")
            
            custom_guidelines = st.text_area(
                "Additional guidelines to include:",
                value=st.session_state.custom_guidelines_text,
                height=150,
                placeholder="e.g., Flag posts about political campaigns during election season\nFlag posts containing investment advice\nFlag posts with malicious links or scams",
                help="These guidelines will be added to the default moderation policy."
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save Guidelines", type="primary"):
                    st.session_state.custom_guidelines_text = custom_guidelines
                    st.session_state.active_policy_type = "add_guidelines"
                    st.success("✅ Custom guidelines saved!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Back to Default"):
                    st.session_state.active_policy_type = "default"
                    st.success("✅ Switched back to default policy!")
                    st.rerun()
        
        else:
            # Default policy selected
            if st.session_state.active_policy_type != "default":
                st.session_state.active_policy_type = "default"
                st.rerun()
        
        # Show current policy status
        st.divider()
        if st.session_state.active_policy_type == "custom_policy" and st.session_state.get("custom_policy_text", "").strip():
            st.info("**Active Policy:** Custom Policy")
            with st.expander("View Custom Policy"):
                st.write(st.session_state.custom_policy_text)
        elif st.session_state.active_policy_type == "add_guidelines" and st.session_state.custom_guidelines_text.strip():
            st.info("**Active Policy:** Default + Custom Guidelines")
            with st.expander("View Custom Guidelines"):
                st.write(st.session_state.custom_guidelines_text)
        else:
            st.info("**Active Policy:** Default - Anti-Misogyny & Anti-Transphobia")
    
    with tab1:
        # Add Quick Stats at the top in a compact horizontal layout
        # Initialize stats if not exists
        if "total_moderated" not in st.session_state:
            st.session_state.total_moderated = 0
            st.session_state.allowed_count = 0
            st.session_state.flagged_count = 0
            st.session_state.review_count = 0
        
        # Compact stats in one line
        st.markdown("**📊 Stats:** " + 
                   f"Total: {st.session_state.total_moderated} | " +
                   f"✅ {st.session_state.allowed_count} | " + 
                   f"🚫 {st.session_state.flagged_count} | " +
                   f"⚠️ {st.session_state.review_count}")
        
        st.markdown("---")
        
        # Main content layout
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
            
            # Jetstream data buttons  
            live_col1, live_col2, live_col3 = st.columns(3)
            
            with live_col1:
                live_stream_button = st.button("🔴 Start Jetstream", use_container_width=True)
            
            with live_col2:
                live_demo_button = st.button("📡 Moderate Live Posts", use_container_width=True)
            
            with live_col3:
                stop_stream_button = st.button("⏹️ Stop Jetstream", use_container_width=True)
            
            # Add debug button
            debug_button = st.button("🔧 Debug Connection", use_container_width=True)
        
        with col2:
            st.header("📡 Live Data & Controls")
            
            # Initialize Jetstream state
            initialize_jetstream_state()
            
            st.markdown("**Connection Status:**")
            
            # Initialize agentic labeler
            if "agentic_labeler" not in st.session_state:
                st.session_state.agentic_labeler = create_demo_labeler(dry_run=True)
                st.session_state.agentic_moderator = AgenticModerator(st.session_state.agentic_labeler)
                st.session_state.agentic_enabled = False
            
            # Initialize feedback system
            if "feedback_system" not in st.session_state:
                st.session_state.feedback_system = FeedbackSystem()
                st.session_state.feedback_analyzer = FeedbackAnalyzer(st.session_state.feedback_system)
            
            # Initialize guidelines manager
            if "guidelines_manager" not in st.session_state:
                st.session_state.guidelines_manager = GuidelinesManager()
            
            # Display connection status with more detail
            is_connected = st.session_state.get("jetstream_connected", False)
            has_client = st.session_state.get("jetstream_client") is not None
            
            if is_connected and has_client:
                st.success("🟢 Connected to Jetstream")
                st.caption("Receiving live AT Protocol posts")
            elif has_client and not is_connected:
                st.warning("🟡 Jetstream client running but not connected")
                st.caption("Client exists but connection lost")
            else:
                st.error("🔴 Not connected to Jetstream")
                st.caption("Click 'Start Jetstream' to begin streaming")
            
            # Live stream metrics - always show
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Live Posts", st.session_state.get("total_live_posts", 0))
            with col_m2:
                queue_size = st.session_state.jetstream_queue.qsize() if "jetstream_queue" in st.session_state else 0
                st.metric("Queue", queue_size)
            
            # Auto-refresh when Jetstream is connected
            if st.session_state.get("jetstream_connected", False):
                # Add auto-refresh every 2 seconds when streaming
                if st.button("🔄 Refresh Metrics", key="refresh_metrics"):
                    st.rerun()
                st.caption("Metrics auto-refresh when streaming is active")
                # Force a rerun every few seconds when connected
                if "last_refresh" not in st.session_state:
                    st.session_state.last_refresh = time.time()
                elif time.time() - st.session_state.last_refresh > 3:  # Refresh every 3 seconds
                    st.session_state.last_refresh = time.time()
                    st.rerun()
            else:
                # Manual refresh when not streaming
                if st.button("🔄 Refresh Metrics", key="refresh_metrics_manual"):
                    st.rerun()
            
            # Advanced features in compact form
            st.divider()
            st.markdown("**Advanced Features:**")
            
            # Agentic labeling - compact
            agentic_enabled = st.checkbox(
                "🤖 Auto-label flagged content", 
                value=st.session_state.get("agentic_enabled", False)
            )
            st.session_state.agentic_enabled = agentic_enabled
            
            # Compact stats display
            with st.expander("📊 System Stats", expanded=False):
                # Agentic stats
                if "agentic_labeler" in st.session_state:
                    stats = st.session_state.agentic_labeler.get_stats()
                    st.write(f"**Labels Assigned:** {stats.get('labels_assigned', 0)}")
                    st.write(f"**Pending Review:** {stats.get('labels_queued_for_review', 0)}")
                
                # Feedback stats
                if "feedback_system" in st.session_state:
                    feedback_stats = st.session_state.feedback_system.get_feedback_stats()
                    st.write(f"**Total Feedback:** {feedback_stats.get('total_submissions', 0)}")
                    if feedback_stats.get('total_submissions', 0) > 0:
                        st.write(f"**Avg Confidence:** {feedback_stats.get('avg_confidence', 0):.1f}/5")
                else:
                    st.info("System initializing...")
    
        # Handle button actions
        if clear_button:
            st.rerun()
        
        if debug_button:
            test_ollama_connection()
    
        # Jetstream button handlers
        if live_stream_button:
            try:
                start_jetstream_monitoring()
                st.success("🔴 Jetstream started! Posts will appear in the queue.")
                # Force a rerun to update the connection status immediately
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start Jetstream: {e}")
        
        if stop_stream_button:
            try:
                stop_jetstream_monitoring()
                st.info("⏹️ Jetstream stopped.")
                # Force a rerun to update the connection status immediately
                st.rerun()
            except Exception as e:
                st.error(f"Failed to stop Jetstream: {e}")
        
        if live_demo_button:
            st.divider()
            st.header("📡 Live AT Protocol Data Moderation")
            
            # Get live posts from queue
            live_posts = get_live_posts(10)
            
            if not live_posts:
                if not st.session_state.get("jetstream_connected", False):
                    st.warning("⚠️ Not connected to Jetstream. Click 'Start Jetstream' first.")
                else:
                    st.info("🔄 No new posts available. Posts are coming in real-time - try again in a moment.")
            else:
                st.success(f"🎯 Processing {len(live_posts)} live posts from AT Protocol")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each live post
                for i, post_data in enumerate(live_posts):
                    status_text.text(f"Processing live post {i+1} of {len(live_posts)}...")
                    progress_bar.progress((i + 1) / len(live_posts))
                    
                    post_text = post_data.get("text", "")
                    author_did = post_data.get("author_did", "unknown")
                    created_at = post_data.get("created_at", "")
                    
                    with st.expander(f"Live Post {i+1}: {post_text[:50]}...", expanded=(i==0)):
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.markdown("**Live AT Protocol Post:**")
                            st.write(post_text)
                        
                        with col_b:
                            st.markdown("**Author DID:**")
                            st.code(author_did[-20:] + "..." if len(author_did) > 20 else author_did)
                            st.markdown("**Created:**")
                            st.write(created_at[:19] if created_at else "Unknown")
                        
                        with st.spinner("Moderating live post..."):
                            # Create AT URI for live post
                            post_uri = f"at://{author_did}/app.bsky.feed.post/{post_data.get('uri', 'unknown')}"
                            
                            moderation_result = moderate_post(post_text, post_uri)
                            
                            # Process with agentic labeling if enabled
                            enhanced_result = process_with_agentic_labeling(moderation_result, post_uri)
                            
                            # Update stats
                            st.session_state.total_moderated += 1
                            if enhanced_result["classification"] == "Allowed":
                                st.session_state.allowed_count += 1
                            elif enhanced_result["classification"] == "Flagged":
                                st.session_state.flagged_count += 1
                            elif enhanced_result["classification"] == "Needs Human Review":
                                st.session_state.review_count += 1
                            
                            display_moderation_result(enhanced_result, post_text, post_uri)
                    
                    # Small delay to prevent overwhelming the API
                    time.sleep(0.5)
                
                status_text.text("✅ Live post moderation complete!")
                st.balloons()
        
        if analyze_button:
            if user_input.strip():
                with st.spinner("🤖 Analyzing content..."):
                    moderation_result = moderate_post(user_input)
                    
                    # Process with agentic labeling if enabled
                    enhanced_result = process_with_agentic_labeling(moderation_result)
                    
                    # Update stats
                    st.session_state.total_moderated += 1
                    if enhanced_result["classification"] == "Allowed":
                        st.session_state.allowed_count += 1
                    elif enhanced_result["classification"] == "Flagged":
                        st.session_state.flagged_count += 1
                    elif enhanced_result["classification"] == "Needs Human Review":
                        st.session_state.review_count += 1
                    
                    st.divider()
                    st.header("🎯 Moderation Result")
                    display_moderation_result(enhanced_result, user_input)
            else:
                st.error("Please enter some content to moderate.")
        
        if demo_button:
            st.divider()
            st.header("🎲 Demo: Moderating Random Posts")
            
            # Load posts from HuggingFace dataset
            dataset_posts = load_sample_posts()
            
            if not dataset_posts:
                st.error("❌ No dataset posts available. Please check your internet connection or try the live stream instead.")
                st.stop()
            
            # Select 10 random posts
            selected_posts = random.sample(dataset_posts, min(10, len(dataset_posts)))
            
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
                        moderation_result = moderate_post(post)
                        
                        # Process with agentic labeling if enabled
                        enhanced_result = process_with_agentic_labeling(moderation_result)
                        
                        # Update stats
                        st.session_state.total_moderated += 1
                        if enhanced_result["classification"] == "Allowed":
                            st.session_state.allowed_count += 1
                        elif enhanced_result["classification"] == "Flagged":
                            st.session_state.flagged_count += 1
                        elif enhanced_result["classification"] == "Needs Human Review":
                            st.session_state.review_count += 1
                        
                        display_moderation_result(enhanced_result, post)
                
                # Small delay to prevent overwhelming the API
                time.sleep(0.5)
            
            status_text.text("✅ Demo complete!")
            st.balloons()
    
    # Feedback Analytics Section
    if st.session_state.get("feedback_system") and st.session_state.feedback_system.get_feedback_stats()["total_submissions"] > 0:
        st.divider()
        st.header("📊 Feedback Analytics")
        
        with st.expander("View Detailed Feedback Analysis"):
            analyzer = st.session_state.feedback_analyzer
            analysis = analyzer.analyze_patterns()
            
            if "error" not in analysis:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("False Positive Rate", f"{analysis.get('false_positive_rate', 0):.1%}")
                    st.metric("False Negative Rate", f"{analysis.get('false_negative_rate', 0):.1%}")
                
                with col2:
                    st.write("**Most Problematic Labels:**")
                    for label, count in list(analysis.get("most_problematic_labels", {}).items())[:3]:
                        st.write(f"• {label}: {count} issues")
                
                with col3:
                    st.write("**Most Missed Labels:**")
                    for label, count in list(analysis.get("most_missed_labels", {}).items())[:3]:
                        st.write(f"• {label}: {count} missed")
                
                # Improvement suggestions
                suggestions = analyzer.get_improvement_suggestions()
                if suggestions:
                    st.subheader("💡 AI Improvement Suggestions")
                    for i, suggestion in enumerate(suggestions, 1):
                        st.write(f"{i}. {suggestion}")
                
                # Recent feedback
                recent_feedback = st.session_state.feedback_system.get_recent_feedback(5)
                if recent_feedback:
                    st.subheader("📝 Recent Feedback")
                    for feedback in recent_feedback:
                        with st.expander(f"{feedback.feedback_type.value.title()} - {feedback.submitted_at.strftime('%m/%d %H:%M')}"):
                            st.write(f"**Post:** {feedback.post_text[:100]}...")
                            st.write(f"**Original:** {feedback.original_classification}")
                            st.write(f"**Correct:** {feedback.correct_classification}")
                            st.write(f"**Explanation:** {feedback.user_explanation}")
                            st.write(f"**Confidence:** {feedback.confidence_in_feedback}/5")
            else:
                st.info("Not enough feedback data for analysis yet.")
    
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
