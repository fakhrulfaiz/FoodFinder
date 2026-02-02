"""
FoodFinder Chat App - Streamlit Frontend
A ChatGPT-like interface for the FoodFinder ReAct agent
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env FIRST
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "FoodFinder"

# Verify setup (only show in terminal, not in Streamlit UI)
if not st.session_state.get("_env_verified", False):
    print("✅ Environment loaded")
    print(f"   LangSmith Tracing: {os.environ.get('LANGSMITH_TRACING')}")
    print(f"   Project: {os.environ.get('LANGSMITH_PROJECT')}")
    api_key = os.environ.get('LANGSMITH_API_KEY', 'NOT SET')
    if api_key != 'NOT SET':
        print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    else:
        print(f"   API Key: NOT SET")
    st.session_state["_env_verified"] = True

# Import agent after environment is loaded
from src.agent.main_agent import FoodFinderAgent

# Page config
st.set_page_config(
    page_title="FoodFinder Chat",
    page_icon="🍽️",
    layout="centered"
)

# Title
st.title("🍽️ FoodFinder Chat")
st.caption("Powered by ReAct Agent with RAG")

# Initialize the agent (cached to avoid reloading)
@st.cache_resource
def get_agent():
    """Initialize and cache the FoodFinder agent"""
    return FoodFinderAgent()

agent = get_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize uploaded image path
if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None

# Create temp directory for uploaded images
temp_dir = project_root / "data" / ".temp"
temp_dir.mkdir(parents=True, exist_ok=True)

# Image upload section (above chat)
uploaded_file = st.file_uploader(
    "📸 Upload a food/restaurant image (optional)",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to search for similar restaurants or ask questions about it"
)

# Handle image upload
if uploaded_file is not None:
    # Clean up old temp files (older than 1 hour)
    import time
    current_time = time.time()
    for old_file in temp_dir.glob("upload_*"):
        if current_time - old_file.stat().st_mtime > 3600:  # 1 hour
            try:
                old_file.unlink()
            except:
                pass
    
    # Save uploaded file to temp directory
    timestamp = int(current_time)
    file_extension = uploaded_file.name.split(".")[-1]
    temp_image_path = temp_dir / f"upload_{timestamp}.{file_extension}"
    
    # Write the uploaded file
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Store path in session state
    st.session_state.uploaded_image_path = str(temp_image_path)
    
    # # Display the uploaded image
    # st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.success(f"Image uploaded! You can now ask questions about it or search for similar restaurants.")
    st.info(f"💡 Try: 'What type of cuisine is this?' or 'Find similar restaurants'")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display image if it was part of the message
        if "image_path" in message:
            st.image(message["image_path"], width=300)

# Accept user input
if prompt := st.chat_input("Ask me about restaurants..."):
    # Prepare user message with optional image
    user_message = {"role": "user", "content": prompt}
    
    # If there's an uploaded image, include it in the message
    if st.session_state.uploaded_image_path:
        user_message["image_path"] = st.session_state.uploaded_image_path
        # Enhance prompt with image context
        enhanced_prompt = f"{prompt}\n\n[Image uploaded: {st.session_state.uploaded_image_path}]"
    else:
        enhanced_prompt = prompt
    
    # Add user message to chat history
    st.session_state.messages.append(user_message)
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        # Display image if uploaded
        if st.session_state.uploaded_image_path:
            st.image(st.session_state.uploaded_image_path, width=300)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Create a placeholder for streaming
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the agent's response
        try:
            # Use the agent's stream method to get updates (with enhanced prompt)
            for chunk in agent.stream(enhanced_prompt):
                # Extract the latest message from the chunk
                if "agent" in chunk:
                    messages = chunk["agent"]["messages"]
                    if messages:
                        latest_message = messages[-1]
                        # Check if it's an AI message (not a tool call)
                        if hasattr(latest_message, 'content') and latest_message.content:
                            full_response = latest_message.content
                            message_placeholder.markdown(full_response + "▌")
            
            # Display final response
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            message_placeholder.markdown(error_msg)
            full_response = error_msg
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Clear uploaded image after processing
    st.session_state.uploaded_image_path = None

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is a **ReAct agent** for restaurant recommendations.
    
    **Available Tools:**
    - 🔍 Text Search - Find restaurants by cuisine, location, etc.
    - 🖼️ Image Search - Find similar restaurants by image
    - 🤔 Image Q&A - Ask questions about food images
    
    **How it works:**
    1. **Reason** - Agent thinks about what info it needs
    2. **Act** - Agent calls appropriate tools
    3. **Observe** - Agent reviews the results
    4. **Repeat** - Until it can answer your question
    """)
    
    st.divider()
    
    st.markdown("**Tips:**")
    st.markdown("""
    **Text queries:**
    - "Find Italian restaurants in Philadelphia"
    - "What are the best pizza places?"
    - "Show me highly rated sushi restaurants"
    
    **With images:**
    - Upload an image, then ask: "What type of cuisine is this?"
    - Upload an image, then ask: "Find similar restaurants"
    - Upload an image, then ask: "Describe this dish"
    """)
    
    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
