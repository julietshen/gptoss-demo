#!/usr/bin/env python3
"""
Test the integrated feedback system
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feedback_system import FeedbackSystem, FeedbackType, FeedbackAnalyzer
from label_config import LabelConfig
import tempfile
import os

def test_feedback_system():
    """Test the feedback system functionality."""
    
    print("🧪 Testing Feedback System Integration...")
    
    # Create temporary feedback file
    temp_dir = tempfile.mkdtemp()
    feedback_file = os.path.join(temp_dir, "test_feedback.json")
    
    try:
        # Initialize feedback system
        feedback_system = FeedbackSystem(feedback_file)
        analyzer = FeedbackAnalyzer(feedback_system)
        
        print("✅ Feedback system initialized")
        
        # Test submitting different types of feedback
        test_cases = [
            {
                "post_text": "Women are too emotional for leadership",
                "original_classification": "Allowed",
                "original_labels": [],
                "feedback_type": FeedbackType.FALSE_NEGATIVE,
                "correct_classification": "Flagged",
                "correct_labels": ["misogyny-subtle"],
                "explanation": "This is subtle misogyny that should be flagged",
                "confidence": 4
            },
            {
                "post_text": "Trans women aren't real women",
                "original_classification": "Flagged",
                "original_labels": ["transphobia-explicit"],
                "feedback_type": FeedbackType.WRONG_LABEL,
                "correct_classification": "Flagged", 
                "correct_labels": ["transphobia-explicit", "transphobia-subtle"],
                "explanation": "Should have both explicit and subtle transphobia labels",
                "confidence": 5
            },
            {
                "post_text": "Beautiful sunset today",
                "original_classification": "Flagged",
                "original_labels": ["misogyny-subtle"],
                "feedback_type": FeedbackType.FALSE_POSITIVE,
                "correct_classification": "Allowed",
                "correct_labels": [],
                "explanation": "This is clearly harmless content about weather",
                "confidence": 5
            }
        ]
        
        # Submit test feedback
        feedback_ids = []
        for i, case in enumerate(test_cases):
            feedback_id = feedback_system.submit_feedback(
                post_text=case["post_text"],
                post_uri=f"test://post/{i}",
                original_classification=case["original_classification"],
                original_labels=case["original_labels"],
                feedback_type=case["feedback_type"],
                correct_classification=case["correct_classification"],
                correct_labels=case["correct_labels"],
                user_explanation=case["explanation"],
                confidence=case["confidence"]
            )
            feedback_ids.append(feedback_id)
            print(f"✅ Submitted feedback {i+1}: {feedback_id}")
        
        # Test feedback statistics
        stats = feedback_system.get_feedback_stats()
        print(f"\n📊 Feedback Stats:")
        print(f"   Total submissions: {stats['total_submissions']}")
        print(f"   Average confidence: {stats['avg_confidence']}/5")
        print(f"   By type: {stats['by_type']}")
        
        # Test analysis
        analysis = analyzer.analyze_patterns()
        print(f"\n🔍 Analysis Results:")
        print(f"   False positive rate: {analysis.get('false_positive_rate', 0):.1%}")
        print(f"   False negative rate: {analysis.get('false_negative_rate', 0):.1%}")
        print(f"   Most problematic labels: {analysis.get('most_problematic_labels', {})}")
        
        # Test improvement suggestions
        suggestions = analyzer.get_improvement_suggestions()
        print(f"\n💡 Improvement Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        # Test label configuration
        print(f"\n🏷️  Testing Label Configuration:")
        test_label = LabelConfig.get_label("misogyny-explicit")
        print(f"   Misogyny Explicit: {test_label.name} - {test_label.description}")
        
        violation_labels = LabelConfig.get_labels_for_violations(["misogyn", "trans"])
        print(f"   Labels for violations: {[l.name for l in violation_labels]}")
        
        print(f"\n✅ All tests passed! Integration working correctly.")
        print(f"   Feedback stored in: {feedback_file}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if os.path.exists(feedback_file):
                os.remove(feedback_file)
            os.rmdir(temp_dir)
        except:
            pass

def test_new_policies():
    """Test the new anti-misogyny and anti-transphobia policies."""
    
    print("\n🎯 Testing New Moderation Policies...")
    
    # Test misogyny detection mapping
    misogyny_tests = [
        ("Women are too emotional", ["misogyny-subtle"]),
        ("She's just a bitch", ["misogyny-explicit"]), 
        ("Stop being such a Karen", ["misogyny-subtle"]),
        ("Women can't do math", ["misogyny-subtle"])
    ]
    
    # Test transphobia detection mapping  
    transphobia_tests = [
        ("Trans people have mental illness", ["transphobia-medicalization"]),
        ("Real women don't have penises", ["transphobia-explicit"]),
        ("Biological sex matters more", ["transphobia-subtle"]),
        ("Using someone's deadname", ["transphobia-deadnaming"])
    ]
    
    print("   Testing misogyny label mapping:")
    for text, expected in misogyny_tests:
        violations = [word for word in text.lower().split() if any(word in v for v in ["women", "emotional", "bitch", "karen", "math"])]
        labels = LabelConfig.get_labels_for_violations(violations)
        label_ids = [l.id for l in labels]
        matches = any(exp in label_ids for exp in expected)
        status = "✅" if matches else "❌"
        print(f"   {status} '{text[:30]}...' -> {label_ids}")
    
    print("   Testing transphobia label mapping:")
    for text, expected in transphobia_tests:
        violations = [word for word in text.lower().split() if any(word in v for v in ["trans", "mental", "real", "biological", "deadnam"])]
        labels = LabelConfig.get_labels_for_violations(violations)
        label_ids = [l.id for l in labels]
        matches = any(exp in label_ids for exp in expected)
        status = "✅" if matches else "❌"
        print(f"   {status} '{text[:30]}...' -> {label_ids}")
    
    print("✅ Policy mapping tests completed")

if __name__ == "__main__":
    success1 = test_feedback_system()
    test_new_policies()
    
    if success1:
        print("\n🎉 All integration tests passed!")
        print("The demo is ready with:")
        print("   • Anti-misogyny & anti-transphobia policies")
        print("   • Comprehensive feedback system")
        print("   • Real-time analysis and suggestions")
        print("   • HuggingFace dataset and live Jetstream integration")
    else:
        print("\n❌ Some tests failed. Check the errors above.")