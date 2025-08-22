"""
Feedback System for Moderation False Positives and Negatives
Allows users to report when the AI makes incorrect decisions
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback that can be submitted."""
    FALSE_POSITIVE = "false_positive"  # AI flagged content that should be allowed
    FALSE_NEGATIVE = "false_negative"  # AI missed content that should be flagged
    WRONG_LABEL = "wrong_label"       # AI applied incorrect label
    SEVERITY_WRONG = "severity_wrong"  # AI got severity level wrong

@dataclass
class FeedbackSubmission:
    """Represents a user feedback submission."""
    feedback_id: str
    post_text: str
    post_uri: str
    original_classification: str
    original_labels: List[str]
    feedback_type: FeedbackType
    correct_classification: str
    correct_labels: List[str]
    user_explanation: str
    confidence_in_feedback: int  # 1-5 scale
    submitted_at: datetime
    reviewed: bool = False
    reviewer_notes: str = ""
    policy_update_needed: bool = False
    
class FeedbackSystem:
    """System for collecting and managing moderation feedback."""
    
    def __init__(self, feedback_file: str = "moderation_feedback.json"):
        """
        Initialize the feedback system.
        
        Args:
            feedback_file: File to store feedback submissions
        """
        self.feedback_file = feedback_file
        self.submissions = []
        self.load_feedback()
    
    def load_feedback(self):
        """Load existing feedback from file."""
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    # Convert datetime string back to datetime object
                    item['submitted_at'] = datetime.fromisoformat(item['submitted_at'])
                    item['feedback_type'] = FeedbackType(item['feedback_type'])
                    self.submissions.append(FeedbackSubmission(**item))
            logger.info(f"Loaded {len(self.submissions)} feedback submissions")
        except FileNotFoundError:
            logger.info("No existing feedback file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
    
    def save_feedback(self):
        """Save feedback submissions to file."""
        try:
            # Convert submissions to dict format for JSON serialization
            data = []
            for submission in self.submissions:
                item = asdict(submission)
                item['submitted_at'] = submission.submitted_at.isoformat()
                item['feedback_type'] = submission.feedback_type.value
                data.append(item)
            
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.submissions)} feedback submissions")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def submit_feedback(self,
                       post_text: str,
                       post_uri: str,
                       original_classification: str,
                       original_labels: List[str],
                       feedback_type: FeedbackType,
                       correct_classification: str,
                       correct_labels: List[str],
                       user_explanation: str,
                       confidence: int) -> str:
        """
        Submit new feedback about a moderation decision.
        
        Returns:
            Feedback ID for the submission
        """
        feedback_id = f"fb_{int(time.time())}_{len(self.submissions)}"
        
        submission = FeedbackSubmission(
            feedback_id=feedback_id,
            post_text=post_text,
            post_uri=post_uri,
            original_classification=original_classification,
            original_labels=original_labels,
            feedback_type=feedback_type,
            correct_classification=correct_classification,
            correct_labels=correct_labels,
            user_explanation=user_explanation,
            confidence_in_feedback=confidence,
            submitted_at=datetime.now()
        )
        
        self.submissions.append(submission)
        self.save_feedback()
        
        logger.info(f"New feedback submitted: {feedback_id} - {feedback_type.value}")
        return feedback_id
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback submissions."""
        if not self.submissions:
            return {
                "total_submissions": 0,
                "by_type": {},
                "avg_confidence": 0,
                "reviewed_count": 0,
                "policy_updates_needed": 0
            }
        
        by_type = {}
        for submission in self.submissions:
            feedback_type = submission.feedback_type.value
            by_type[feedback_type] = by_type.get(feedback_type, 0) + 1
        
        avg_confidence = sum(s.confidence_in_feedback for s in self.submissions) / len(self.submissions)
        reviewed_count = sum(1 for s in self.submissions if s.reviewed)
        policy_updates_needed = sum(1 for s in self.submissions if s.policy_update_needed)
        
        return {
            "total_submissions": len(self.submissions),
            "by_type": by_type,
            "avg_confidence": round(avg_confidence, 2),
            "reviewed_count": reviewed_count,
            "policy_updates_needed": policy_updates_needed
        }
    
    def get_recent_feedback(self, limit: int = 10) -> List[FeedbackSubmission]:
        """Get most recent feedback submissions."""
        return sorted(self.submissions, key=lambda x: x.submitted_at, reverse=True)[:limit]
    
    def get_false_positives(self) -> List[FeedbackSubmission]:
        """Get all false positive feedback."""
        return [s for s in self.submissions if s.feedback_type == FeedbackType.FALSE_POSITIVE]
    
    def get_false_negatives(self) -> List[FeedbackSubmission]:
        """Get all false negative feedback."""
        return [s for s in self.submissions if s.feedback_type == FeedbackType.FALSE_NEGATIVE]
    
    def mark_reviewed(self, feedback_id: str, reviewer_notes: str = "", policy_update_needed: bool = False):
        """Mark feedback as reviewed."""
        for submission in self.submissions:
            if submission.feedback_id == feedback_id:
                submission.reviewed = True
                submission.reviewer_notes = reviewer_notes
                submission.policy_update_needed = policy_update_needed
                self.save_feedback()
                return True
        return False
    
    def export_for_analysis(self) -> List[Dict]:
        """Export feedback data for analysis."""
        export_data = []
        for submission in self.submissions:
            export_data.append({
                "feedback_id": submission.feedback_id,
                "post_text": submission.post_text,
                "original_classification": submission.original_classification,
                "original_labels": submission.original_labels,
                "feedback_type": submission.feedback_type.value,
                "correct_classification": submission.correct_classification,
                "correct_labels": submission.correct_labels,
                "user_explanation": submission.user_explanation,
                "confidence": submission.confidence_in_feedback,
                "submitted_at": submission.submitted_at.isoformat(),
                "reviewed": submission.reviewed
            })
        return export_data

class FeedbackAnalyzer:
    """Analyzes feedback to identify patterns and improvement opportunities."""
    
    def __init__(self, feedback_system: FeedbackSystem):
        self.feedback_system = feedback_system
    
    def analyze_patterns(self) -> Dict:
        """Analyze feedback patterns to identify systematic issues."""
        submissions = self.feedback_system.submissions
        
        if not submissions:
            return {"error": "No feedback data available"}
        
        # Analyze false positive patterns
        false_positives = [s for s in submissions if s.feedback_type == FeedbackType.FALSE_POSITIVE]
        fp_labels = {}
        for fp in false_positives:
            for label in fp.original_labels:
                fp_labels[label] = fp_labels.get(label, 0) + 1
        
        # Analyze false negative patterns
        false_negatives = [s for s in submissions if s.feedback_type == FeedbackType.FALSE_NEGATIVE]
        fn_missed_labels = {}
        for fn in false_negatives:
            for label in fn.correct_labels:
                fn_missed_labels[label] = fn_missed_labels.get(label, 0) + 1
        
        # Calculate accuracy by label type
        label_accuracy = {}
        all_labels = set()
        for s in submissions:
            all_labels.update(s.original_labels)
            all_labels.update(s.correct_labels)
        
        for label in all_labels:
            correct = sum(1 for s in submissions if label in s.original_labels and label in s.correct_labels)
            total = sum(1 for s in submissions if label in s.original_labels or label in s.correct_labels)
            if total > 0:
                label_accuracy[label] = round(correct / total, 2)
        
        return {
            "total_feedback": len(submissions),
            "false_positive_rate": len(false_positives) / len(submissions) if submissions else 0,
            "false_negative_rate": len(false_negatives) / len(submissions) if submissions else 0,
            "most_problematic_labels": dict(sorted(fp_labels.items(), key=lambda x: x[1], reverse=True)[:5]),
            "most_missed_labels": dict(sorted(fn_missed_labels.items(), key=lambda x: x[1], reverse=True)[:5]),
            "label_accuracy": label_accuracy,
            "high_confidence_feedback": len([s for s in submissions if s.confidence_in_feedback >= 4]),
            "needs_policy_update": len([s for s in submissions if s.policy_update_needed])
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """Generate suggestions for improving moderation accuracy."""
        analysis = self.analyze_patterns()
        suggestions = []
        
        if analysis.get("false_positive_rate", 0) > 0.2:
            suggestions.append("High false positive rate detected. Consider raising confidence thresholds for auto-assignment.")
        
        if analysis.get("false_negative_rate", 0) > 0.2:
            suggestions.append("High false negative rate detected. Consider improving prompt specificity or lowering detection thresholds.")
        
        problematic_labels = analysis.get("most_problematic_labels", {})
        if problematic_labels:
            top_problem = list(problematic_labels.keys())[0]
            suggestions.append(f"Label '{top_problem}' has high false positive rate. Review its detection criteria.")
        
        missed_labels = analysis.get("most_missed_labels", {})
        if missed_labels:
            top_missed = list(missed_labels.keys())[0]
            suggestions.append(f"Label '{top_missed}' is frequently missed. Enhance prompt to better detect this category.")
        
        if analysis.get("needs_policy_update", 0) > 0:
            suggestions.append("Some feedback indicates policy updates are needed. Review flagged submissions.")
        
        return suggestions[:5]  # Return top 5 suggestions