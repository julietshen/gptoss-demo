"""
AT Protocol Labeler Client
Handles authentication and label assignment for Bluesky posts
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from dataclasses import dataclass, asdict

from label_config import ModerationLabel, LabelConfig, AgenticPolicy

logger = logging.getLogger(__name__)

@dataclass
class LabelAssignment:
    """Represents a label assignment for a post or account."""
    uri: str                    # AT URI of the content
    label_id: str              # Label identifier
    confidence: float          # AI confidence score (0.0-1.0)
    reasoning: str             # Human-readable reasoning
    created_at: datetime       # When label was assigned
    assigned_by: str           # "ai-agent" or "human"
    reviewed: bool = False     # Whether human has reviewed
    
class ATLabelerClient:
    """Client for assigning labels via AT Protocol."""
    
    def __init__(self, 
                 did: str,
                 signing_key: str, 
                 service_endpoint: str = "https://bsky.social",
                 dry_run: bool = True):
        """
        Initialize the labeler client.
        
        Args:
            did: Decentralized identifier for the labeler
            signing_key: Private key for signing requests
            service_endpoint: AT Protocol service endpoint
            dry_run: If True, don't actually assign labels (for testing)
        """
        self.did = did
        self.signing_key = signing_key
        self.service_endpoint = service_endpoint
        self.dry_run = dry_run
        
        # Rate limiting
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        
        # Statistics
        self.stats = {
            "labels_assigned": 0,
            "labels_queued_for_review": 0,
            "api_calls": 0,
            "rate_limited": 0
        }
        
        logger.info(f"Initialized AT Labeler Client (DID: {did}, Dry Run: {dry_run})")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        
        # Reset counter if a new minute has started
        if current_time - self.minute_start > 60:
            self.minute_start = current_time
            self.requests_this_minute = 0
        
        # Check if we're under the limit
        if self.requests_this_minute >= AgenticPolicy.RATE_LIMIT_PER_MINUTE:
            self.stats["rate_limited"] += 1
            return False
        
        return True
    
    def _make_authenticated_request(self, method: str, endpoint: str, data: dict = None) -> requests.Response:
        """Make an authenticated request to the AT Protocol service."""
        
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        # In a real implementation, this would:
        # 1. Create proper AT Protocol authentication headers
        # 2. Sign the request with the private key
        # 3. Handle DID resolution and verification
        
        # For now, simulate the request
        url = f"{self.service_endpoint}/xrpc/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.signing_key}",  # Simplified
            "Content-Type": "application/json"
        }
        
        if self.dry_run:
            logger.info(f"DRY RUN: {method} {endpoint} with data: {data}")
            # Simulate successful response
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"success": True}).encode()
            return response
        
        self.requests_this_minute += 1
        self.stats["api_calls"] += 1
        
        if method.upper() == "POST":
            return requests.post(url, headers=headers, json=data, timeout=10)
        else:
            return requests.get(url, headers=headers, params=data, timeout=10)
    
    def assign_label(self, 
                    post_uri: str, 
                    label: ModerationLabel, 
                    confidence: float,
                    reasoning: str) -> bool:
        """
        Assign a label to a post.
        
        Args:
            post_uri: AT URI of the post to label
            label: Label configuration to assign
            confidence: AI confidence score
            reasoning: Explanation for the assignment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create label assignment record
            assignment = LabelAssignment(
                uri=post_uri,
                label_id=label.id,
                confidence=confidence,
                reasoning=reasoning,
                created_at=datetime.now(),
                assigned_by="ai-agent"
            )
            
            # Prepare AT Protocol label creation request
            label_data = {
                "repo": self.did,
                "collection": "app.bsky.label",
                "record": {
                    "$type": "app.bsky.label",
                    "uri": post_uri,
                    "val": label.id,
                    "neg": False,  # Positive label (not negating)
                    "cts": assignment.created_at.isoformat(),
                    "exp": None,   # No expiration
                    "sig": None    # Signature (would be added in real implementation)
                }
            }
            
            # Make the API call
            response = self._make_authenticated_request(
                "POST", 
                "com.atproto.repo.createRecord",
                label_data
            )
            
            if response.status_code == 200:
                self.stats["labels_assigned"] += 1
                logger.info(f"Successfully assigned label '{label.id}' to {post_uri} (confidence: {confidence:.2f})")
                return True
            else:
                logger.error(f"Failed to assign label: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error assigning label: {e}")
            return False
    
    def batch_assign_labels(self, assignments: List[Tuple[str, ModerationLabel, float, str]]) -> Dict[str, bool]:
        """
        Assign multiple labels in batch.
        
        Args:
            assignments: List of (post_uri, label, confidence, reasoning) tuples
            
        Returns:
            Dictionary mapping post URIs to success status
        """
        results = {}
        
        for post_uri, label, confidence, reasoning in assignments:
            # Add small delay to respect rate limits
            time.sleep(0.1)
            
            success = self.assign_label(post_uri, label, confidence, reasoning)
            results[post_uri] = success
        
        return results
    
    def query_labels(self, post_uri: str) -> List[Dict]:
        """
        Query existing labels for a post.
        
        Args:
            post_uri: AT URI of the post
            
        Returns:
            List of existing label records
        """
        try:
            response = self._make_authenticated_request(
                "GET",
                "com.atproto.label.queryLabels",
                {"uriPatterns": [post_uri]}
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("labels", [])
            else:
                logger.error(f"Failed to query labels: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error querying labels: {e}")
            return []
    
    def remove_label(self, post_uri: str, label_id: str) -> bool:
        """
        Remove a label from a post.
        
        Args:
            post_uri: AT URI of the post
            label_id: Label to remove
            
        Returns:
            True if successful
        """
        try:
            # In a real implementation, this would create a "neg" label
            # to negate the previous positive label
            neg_label_data = {
                "repo": self.did,
                "collection": "app.bsky.label", 
                "record": {
                    "$type": "app.bsky.label",
                    "uri": post_uri,
                    "val": label_id,
                    "neg": True,  # Negative label (negating previous)
                    "cts": datetime.now().isoformat()
                }
            }
            
            response = self._make_authenticated_request(
                "POST",
                "com.atproto.repo.createRecord", 
                neg_label_data
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error removing label: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get labeler statistics."""
        return self.stats.copy()


class AgenticModerator:
    """Agentic moderation system that automatically assigns labels."""
    
    def __init__(self, labeler_client: ATLabelerClient):
        """
        Initialize the agentic moderator.
        
        Args:
            labeler_client: Configured AT Protocol labeler client
        """
        self.labeler = labeler_client
        self.pending_review = []  # Posts requiring human review
        
    def process_moderation_result(self, 
                                post_uri: str,
                                moderation_result: Dict[str, str]) -> Dict[str, any]:
        """
        Process a moderation result and decide on automatic actions.
        
        Args:
            post_uri: AT URI of the post
            moderation_result: Result from GPT-OSS moderation
            
        Returns:
            Processing result with actions taken
        """
        classification = moderation_result.get("classification", "Unknown")
        violation_type = moderation_result.get("violation_type", "N/A")
        rationale = moderation_result.get("rationale", "")
        
        # Simulate confidence score based on classification
        # In a real system, this would come from the LLM
        confidence = self._estimate_confidence(classification, violation_type, rationale)
        
        result = {
            "post_uri": post_uri,
            "classification": classification,
            "confidence": confidence,
            "labels_assigned": [],
            "action_taken": "none",
            "requires_human_review": False,
            "reasoning": rationale
        }
        
        # Determine appropriate labels
        violation_types = [violation_type] if violation_type != "N/A" else []
        potential_labels = LabelConfig.get_labels_for_violations(violation_types)
        
        # Check if we should escalate to human review
        if AgenticPolicy.should_escalate_to_human(classification, confidence, violation_type):
            result["requires_human_review"] = True
            result["action_taken"] = "escalated-to-human"
            self.pending_review.append(result)
            return result
        
        # Process auto-assignable labels
        if AgenticPolicy.AUTO_ASSIGN_ENABLED and classification == "Flagged":
            auto_labels = []
            
            for label in potential_labels:
                if AgenticPolicy.should_auto_assign(label, confidence):
                    # Only auto-assign "safe" labels if policy requires it
                    if (AgenticPolicy.AUTO_ASSIGN_SAFE_LABELS_ONLY and 
                        label.severity.value in ["block", "hide"]):
                        continue
                    
                    auto_labels.append(label)
            
            # Assign the labels
            for label in auto_labels:
                success = self.labeler.assign_label(
                    post_uri=post_uri,
                    label=label,
                    confidence=confidence,
                    reasoning=f"Automatic assignment: {rationale}"
                )
                
                if success:
                    result["labels_assigned"].append(label.id)
            
            if auto_labels:
                result["action_taken"] = AgenticPolicy.get_recommended_action(
                    classification, confidence, auto_labels
                )
        
        return result
    
    def _estimate_confidence(self, classification: str, violation_type: str, rationale: str) -> float:
        """
        Estimate confidence score based on moderation output.
        This is a simplified heuristic - in practice you'd want the LLM to output confidence directly.
        """
        base_confidence = 0.5
        
        # Adjust based on classification
        if classification == "Allowed":
            base_confidence = 0.8
        elif classification == "Flagged":
            base_confidence = 0.75
        elif classification == "Needs Human Review":
            base_confidence = 0.6
        
        # Adjust based on violation type specificity
        if violation_type != "N/A" and len(violation_type) > 5:
            base_confidence += 0.1
        
        # Adjust based on rationale length/detail
        if len(rationale) > 100:
            base_confidence += 0.05
        
        return min(0.95, max(0.1, base_confidence))
    
    def get_pending_review(self) -> List[Dict]:
        """Get posts pending human review."""
        return self.pending_review.copy()
    
    def clear_pending_review(self):
        """Clear the pending review queue."""
        self.pending_review.clear()


# Example usage and configuration
def create_demo_labeler(dry_run: bool = True) -> ATLabelerClient:
    """Create a demo labeler client for testing."""
    return ATLabelerClient(
        did="did:plc:example123456789",
        signing_key="demo-key-not-real",
        dry_run=dry_run
    )

def demo_agentic_labeling():
    """Demonstrate the agentic labeling workflow."""
    
    # Create labeler and moderator
    labeler = create_demo_labeler(dry_run=True)
    moderator = AgenticModerator(labeler)
    
    # Example moderation results
    test_cases = [
        {
            "uri": "at://did:plc:user1/app.bsky.feed.post/123",
            "result": {
                "classification": "Flagged",
                "violation_type": "Spam",
                "rationale": "Repetitive promotional content with multiple links"
            }
        },
        {
            "uri": "at://did:plc:user2/app.bsky.feed.post/456", 
            "result": {
                "classification": "Flagged",
                "violation_type": "Hate Speech",
                "rationale": "Contains targeted harassment against a protected group"
            }
        },
        {
            "uri": "at://did:plc:user3/app.bsky.feed.post/789",
            "result": {
                "classification": "Allowed",
                "violation_type": "N/A", 
                "rationale": "Content appears to be normal conversation"
            }
        }
    ]
    
    print("🤖 Agentic Labeling Demo")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\nProcessing: {test['uri']}")
        print(f"Classification: {test['result']['classification']}")
        print(f"Violation: {test['result']['violation_type']}")
        
        result = moderator.process_moderation_result(test['uri'], test['result'])
        
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Action: {result['action_taken']}")
        print(f"Labels: {result['labels_assigned']}")
        print(f"Human Review: {result['requires_human_review']}")
    
    print(f"\nLabeler Stats: {labeler.get_stats()}")
    print(f"Pending Review: {len(moderator.get_pending_review())} posts")

if __name__ == "__main__":
    demo_agentic_labeling()