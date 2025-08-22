"""
AT Protocol Label Configuration for Agentic Moderation
Defines labels, thresholds, and policies for automatic assignment
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class LabelSeverity(Enum):
    """Label severity levels for different moderation actions."""
    INFO = "info"          # Informational, no action
    WARN = "warn"          # Warning label
    HIDE = "hide"          # Hide behind warning
    BLUR = "blur"          # Blur content  
    BLOCK = "block"        # Block/remove content

@dataclass
class ModerationLabel:
    """Configuration for a moderation label."""
    id: str
    name: str
    description: str
    severity: LabelSeverity
    auto_assign_threshold: float  # Confidence threshold for auto-assignment (0.0-1.0)
    requires_human_review: bool   # Whether this label always needs human review
    
class LabelConfig:
    """Configuration for AT Protocol moderation labels."""
    
    # Anti-Misogyny & Anti-Transphobia Moderation Labels
    # Based on specialized moderation policies for targeted protection
    LABELS = {
        "misogyny-explicit": ModerationLabel(
            id="misogyny-explicit",
            name="Explicit Misogyny",
            description="Content containing overt, deliberate attacks on women or feminine-coded individuals based on their gender",
            severity=LabelSeverity.HIDE,  # alert -> hide in our system
            auto_assign_threshold=0.85,  # High confidence for explicit attacks
            requires_human_review=True   # Always review explicit attacks
        ),
        
        "misogyny-subtle": ModerationLabel(
            id="misogyny-subtle",
            name="Subtle Misogyny",
            description="Content containing implicit or coded misogynistic attitudes and stereotypes",
            severity=LabelSeverity.WARN,  # inform -> warn in our system
            auto_assign_threshold=0.70,  # Lower threshold for subtle cases
            requires_human_review=False  # Can auto-assign subtle cases
        ),
        
        "misogyny-harassment": ModerationLabel(
            id="misogyny-harassment",
            name="Misogynistic Harassment",
            description="Targeted harassment of individuals based on their gender or perceived gender",
            severity=LabelSeverity.HIDE,  # alert -> hide
            auto_assign_threshold=0.80,
            requires_human_review=True   # Always review harassment
        ),
        
        "transphobia-explicit": ModerationLabel(
            id="transphobia-explicit",
            name="Explicit Transphobia",
            description="Direct attacks on transgender individuals or the transgender community",
            severity=LabelSeverity.HIDE,  # alert -> hide
            auto_assign_threshold=0.85,  # High confidence for explicit attacks
            requires_human_review=True   # Always review explicit attacks
        ),
        
        "transphobia-subtle": ModerationLabel(
            id="transphobia-subtle",
            name="Subtle Transphobia",
            description="Content containing coded or indirect transphobic messaging",
            severity=LabelSeverity.WARN,  # inform -> warn
            auto_assign_threshold=0.70,  # Lower threshold for subtle cases
            requires_human_review=False  # Can auto-assign subtle cases
        ),
        
        "transphobia-deadnaming": ModerationLabel(
            id="transphobia-deadnaming",
            name="Deadnaming",
            description="Deliberate use of a transgender person's former name or identity",
            severity=LabelSeverity.HIDE,  # alert -> hide
            auto_assign_threshold=0.90,  # Very high confidence - specific behavior
            requires_human_review=True   # Always review deadnaming
        ),
        
        "transphobia-medicalization": ModerationLabel(
            id="transphobia-medicalization",
            name="Inappropriate Medicalization",
            description="Inappropriate medicalization or pathologization of transgender identities",
            severity=LabelSeverity.WARN,  # inform -> warn
            auto_assign_threshold=0.75,
            requires_human_review=False  # Can auto-assign medicalization cases
        ),
        
        # Keep some general categories for broader coverage
        "harassment-general": ModerationLabel(
            id="harassment-general",
            name="General Harassment",
            description="Targeted harassment not specifically gendered",
            severity=LabelSeverity.HIDE,
            auto_assign_threshold=0.80,
            requires_human_review=True
        ),
        
        "spam": ModerationLabel(
            id="spam",
            name="Spam",
            description="Repetitive, commercial, or promotional content",
            severity=LabelSeverity.WARN,
            auto_assign_threshold=0.75,
            requires_human_review=False
        )
    }
    
    @classmethod
    def get_label(cls, label_id: str) -> Optional[ModerationLabel]:
        """Get label configuration by ID."""
        return cls.LABELS.get(label_id)
    
    @classmethod
    def get_labels_for_violations(cls, violation_types: List[str]) -> List[ModerationLabel]:
        """Get appropriate labels for detected violation types."""
        labels = []
        
        # Map GPT-OSS violation types to our specialized anti-misogyny/transphobia labels
        violation_mapping = {
            # Misogyny detection
            "misogyn": ["misogyny-explicit"],
            "sexist": ["misogyny-subtle"],
            "women": ["misogyny-subtle"],  # Context-dependent
            "female": ["misogyny-subtle"],  # Context-dependent
            "bitch": ["misogyny-explicit"],
            "whore": ["misogyny-explicit"],
            "slut": ["misogyny-explicit"],
            "karen": ["misogyny-subtle"],
            "emotional": ["misogyny-subtle"],  # When used dismissively
            "hysterical": ["misogyny-explicit"],
            
            # Transphobia detection
            "transphob": ["transphobia-explicit"],
            "transgender": ["transphobia-subtle"],  # Context-dependent
            "trans": ["transphobia-subtle"],  # Context-dependent
            "deadnam": ["transphobia-deadnaming"],
            "biological": ["transphobia-subtle"],  # When used to exclude
            "real women": ["transphobia-explicit"],
            "real men": ["transphobia-explicit"],
            "mental illness": ["transphobia-medicalization"],
            "disorder": ["transphobia-medicalization"],
            "detrans": ["transphobia-medicalization"],
            
            # Intersectional harassment
            "harassment": ["misogyny-harassment", "transphobia-explicit", "harassment-general"],
            "hate": ["misogyny-explicit", "transphobia-explicit"],
            "attack": ["misogyny-harassment", "transphobia-explicit"],
            "threat": ["misogyny-harassment", "transphobia-explicit"],
            
            # General categories
            "spam": ["spam"],
            "commercial": ["spam"],
            "promotional": ["spam"]
        }
        
        for violation in violation_types:
            violation_lower = violation.lower()
            for key, label_ids in violation_mapping.items():
                if key in violation_lower:
                    for label_id in label_ids:
                        if label_id in cls.LABELS:
                            labels.append(cls.LABELS[label_id])
        
        # Remove duplicates by ID
        seen_ids = set()
        unique_labels = []
        for label in labels:
            if label.id not in seen_ids:
                unique_labels.append(label)
                seen_ids.add(label.id)
        return unique_labels
    
    @classmethod
    def should_auto_assign(cls, label: ModerationLabel, confidence: float) -> bool:
        """Determine if a label should be auto-assigned based on confidence."""
        if label.requires_human_review:
            return False
        return confidence >= label.auto_assign_threshold
    
    @classmethod
    def get_auto_assignable_labels(cls, confidence_scores: Dict[str, float]) -> List[ModerationLabel]:
        """Get labels that should be automatically assigned based on confidence scores."""
        auto_labels = []
        
        for label_id, confidence in confidence_scores.items():
            label = cls.get_label(label_id)
            if label and cls.should_auto_assign(label, confidence):
                auto_labels.append(label)
        
        return auto_labels


class AgenticPolicy:
    """Policy configuration for agentic moderation decisions."""
    
    # Global thresholds
    MIN_CONFIDENCE_FOR_ACTION = 0.60      # Minimum confidence to take any action
    HIGH_CONFIDENCE_THRESHOLD = 0.85      # Threshold for high-confidence decisions
    BATCH_SIZE = 10                       # Number of posts to process in batch
    RATE_LIMIT_PER_MINUTE = 60           # API calls per minute limit
    
    # Auto-assignment policies
    AUTO_ASSIGN_ENABLED = True           # Enable automatic label assignment
    AUTO_ASSIGN_SAFE_LABELS_ONLY = True  # Only auto-assign "safe" labels (info, warn)
    REQUIRE_HUMAN_REVIEW_THRESHOLD = 0.95  # Confidence above which human review is still required for severe labels
    
    # Escalation policies  
    ESCALATE_CONTROVERSIAL_CONTENT = True    # Send controversial content for human review
    ESCALATE_AMBIGUOUS_RESULTS = True       # Send low-confidence results for review
    MAX_AUTO_ASSIGNMENTS_PER_HOUR = 100     # Limit auto-assignments per hour
    
    @classmethod
    def should_escalate_to_human(cls, classification: str, confidence: float, violation_type: str) -> bool:
        """Determine if content should be escalated to human review."""
        
        # Always escalate if flagged but low confidence
        if classification == "Flagged" and confidence < cls.HIGH_CONFIDENCE_THRESHOLD:
            return True
        
        # Immediate escalation criteria (from policy)
        immediate_escalation = [
            "threat", "violence", "coordinated", "campaign", "minor", "child",
            "doxx", "personal information"
        ]
        if any(term in violation_type.lower() for term in immediate_escalation):
            return True
        
        # Always escalate explicit misogyny and transphobia
        explicit_violations = [
            "misogyny-explicit", "transphobia-explicit", "misogyny-harassment", 
            "transphobia-deadnaming"
        ]
        if any(explicit in violation_type.lower() for explicit in explicit_violations):
            return True
        
        # Escalate edge cases
        if classification == "Needs Human Review":
            return True
        
        return False
    
    @classmethod
    def get_recommended_action(cls, classification: str, confidence: float, labels: List[ModerationLabel]) -> str:
        """Get recommended action based on classification and labels."""
        
        if not labels:
            return "no-action"
        
        # Find most severe label
        max_severity = max(label.severity for label in labels)
        
        if max_severity == LabelSeverity.BLOCK:
            return "block-content"
        elif max_severity == LabelSeverity.HIDE:
            return "hide-content"
        elif max_severity == LabelSeverity.BLUR:
            return "blur-content" 
        elif max_severity == LabelSeverity.WARN:
            return "add-warning"
        else:
            return "add-label"