#!/usr/bin/env python3
"""
Simple test script for Jetstream connection
"""

import time
import logging
from jetstream_client import create_post_stream_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_jetstream():
    """Test Jetstream connection and post reception."""
    print("🚀 Testing Jetstream connection...")
    
    posts_received = []
    
    def on_post(post_data):
        posts_received.append(post_data)
        print(f"📡 Received post {len(posts_received)}: {post_data.get('text', '')[:100]}...")
        print(f"   Author: {post_data.get('author_did', 'unknown')}")
        print(f"   Created: {post_data.get('created_at', 'unknown')}")
        print("-" * 50)
        
        # Stop after 5 posts for testing
        if len(posts_received) >= 5:
            return
    
    # Create client
    client = create_post_stream_client(on_post)
    
    def on_connect():
        print("✅ Connected to Jetstream!")
    
    def on_disconnect():
        print("❌ Disconnected from Jetstream")
    
    def on_error(error):
        print(f"🚨 Error: {error}")
    
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect  
    client.on_error = on_error
    
    # Start the client
    thread = client.start()
    
    try:
        # Wait for some posts
        print("⏳ Waiting for posts (30 seconds)...")
        timeout = 30
        start_time = time.time()
        
        while len(posts_received) < 5 and (time.time() - start_time) < timeout:
            time.sleep(1)
            if len(posts_received) > 0:
                print(f"📊 Posts received so far: {len(posts_received)}")
        
        if posts_received:
            print(f"🎉 Success! Received {len(posts_received)} posts from AT Protocol")
        else:
            print("⚠️ No posts received. This might be normal if the network is quiet.")
            
    except KeyboardInterrupt:
        print("\n👋 Stopping test...")
    finally:
        client.disconnect()
        print("🛑 Test complete")

if __name__ == "__main__":
    test_jetstream()