"""
Jetstream AT Protocol Client
Connects to Bluesky's Jetstream to stream real-time AT Protocol data
"""

import json
import logging
import time
import threading
from typing import Callable, Optional, List, Dict, Any
import websocket
import zstandard as zstd

logger = logging.getLogger(__name__)

class JetstreamClient:
    """Client for connecting to Bluesky's Jetstream real-time data stream."""
    
    # Official Jetstream endpoints
    ENDPOINTS = [
        "wss://jetstream2.us-east.bsky.network/subscribe",
        "wss://jetstream2.us-west.bsky.network/subscribe",
        "wss://jetstream1.us-east.bsky.network/subscribe", 
        "wss://jetstream1.us-west.bsky.network/subscribe"
    ]
    
    def __init__(self, 
                 endpoint: str = None,
                 wanted_collections: List[str] = None,
                 wanted_dids: List[str] = None,
                 compress: bool = True):
        """
        Initialize Jetstream client.
        
        Args:
            endpoint: WebSocket endpoint URL (defaults to first available)
            wanted_collections: Filter by collection NSIDs (e.g., ['app.bsky.feed.post'])
            wanted_dids: Filter by specific DIDs
            compress: Enable zstd compression
        """
        self.endpoint = endpoint or self.ENDPOINTS[0]
        self.wanted_collections = wanted_collections or []
        self.wanted_dids = wanted_dids or []
        self.compress = compress
        
        self.ws = None
        self.is_connected = False
        self.is_running = False
        self.decompressor = zstd.ZstdDecompressor() if compress else None
        
        # Callbacks
        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        
    def _build_url(self) -> str:
        """Build WebSocket URL with query parameters."""
        params = []
        
        if self.wanted_collections:
            for collection in self.wanted_collections:
                params.append(f"wantedCollections={collection}")
                
        if self.wanted_dids:
            for did in self.wanted_dids:
                params.append(f"wantedDids={did}")
                
        if self.compress:
            params.append("compress=true")
            
        url = self.endpoint
        if params:
            url += "?" + "&".join(params)
            
        return url
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            # Handle different message types
            if isinstance(message, bytes):
                if self.compress and self.decompressor:
                    try:
                        # Try to decompress
                        message = self.decompressor.decompress(message).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Failed to decompress message, trying raw decode: {e}")
                        # Fall back to raw decode
                        try:
                            message = message.decode('utf-8')
                        except Exception as e2:
                            logger.error(f"Failed to decode message: {e2}")
                            return
                else:
                    # No compression, just decode
                    try:
                        message = message.decode('utf-8')
                    except Exception as e:
                        logger.error(f"Failed to decode message: {e}")
                        return
            
            # Parse JSON
            if isinstance(message, str):
                data = json.loads(message)
            else:
                logger.error(f"Unexpected message type: {type(message)}")
                return
            
            # Call user callback
            if self.on_message:
                self.on_message(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Message content: {message[:200]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if self.on_error:
                self.on_error(e)
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        if self.on_error:
            self.on_error(error)
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.is_connected = False
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.on_disconnect:
            self.on_disconnect()
    
    def _on_open(self, ws):
        """Handle WebSocket open."""
        self.is_connected = True
        logger.info("Connected to Jetstream")
        if self.on_connect:
            self.on_connect()
    
    def connect(self):
        """Connect to Jetstream."""
        url = self._build_url()
        logger.info(f"Connecting to Jetstream: {url}")
        
        try:
            self.ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            self.is_running = True
            
        except Exception as e:
            logger.error(f"Failed to create WebSocket connection: {e}")
            raise
    
    def start(self):
        """Start the WebSocket connection in a separate thread."""
        if not self.ws:
            self.connect()
            
        def run():
            try:
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
                if self.on_error:
                    self.on_error(e)
            finally:
                self.is_running = False
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
    
    def disconnect(self):
        """Disconnect from Jetstream."""
        self.is_running = False
        if self.ws:
            self.ws.close()
            self.ws = None
        self.is_connected = False
        logger.info("Disconnected from Jetstream")
    
    def send_config(self, config: Dict[str, Any]):
        """Send configuration update to Jetstream."""
        if self.ws and self.is_connected:
            try:
                self.ws.send(json.dumps(config))
                logger.info(f"Sent config: {config}")
            except Exception as e:
                logger.error(f"Failed to send config: {e}")
        else:
            logger.warning("Cannot send config: not connected")


def filter_posts(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter for Bluesky posts from Jetstream data.
    
    Args:
        data: Raw Jetstream message
        
    Returns:
        Formatted post data if it's a post, None otherwise
    """
    try:
        # Check if this is a commit event for a post
        if (data.get("kind") == "commit" and 
            data.get("commit", {}).get("collection") == "app.bsky.feed.post" and
            data.get("commit", {}).get("operation") == "create"):
            
            commit = data["commit"]
            record = commit.get("record", {})
            
            # Extract post content
            post_data = {
                "text": record.get("text", ""),
                "created_at": record.get("createdAt", ""),
                "author_did": data.get("did", ""),
                "uri": commit.get("rkey", ""),
                "language": record.get("langs", ["en"])[0] if record.get("langs") else "en",
                "reply_parent": record.get("reply", {}).get("parent", {}).get("uri") if record.get("reply") else None,
                "embed_type": record.get("embed", {}).get("$type") if record.get("embed") else None,
                "raw_record": record
            }
            
            return post_data
            
    except Exception as e:
        logger.error(f"Error filtering post: {e}")
        
    return None


# Example usage functions
def create_post_stream_client(on_post_callback: Callable[[Dict[str, Any]], None]) -> JetstreamClient:
    """
    Create a Jetstream client configured to stream Bluesky posts.
    
    Args:
        on_post_callback: Function to call when a new post is received
        
    Returns:
        Configured JetstreamClient
    """
    def handle_message(data):
        post_data = filter_posts(data)
        if post_data and post_data.get("text"):
            on_post_callback(post_data)
    
    # Disable compression to avoid decompression errors
    client = JetstreamClient(
        wanted_collections=["app.bsky.feed.post"],
        compress=False
    )
    client.on_message = handle_message
    
    return client