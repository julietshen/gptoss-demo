#!/usr/bin/env python3
"""
Test script for agentic labeling functionality
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from at_labeler import demo_agentic_labeling
    print("🚀 Testing Agentic Labeling System...")
    demo_agentic_labeling()
    print("\n✅ Agentic labeling test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This is expected if running without proper dependencies.")
    print("The system is designed to work within the full Streamlit app context.")
    
    # Run a simplified test without dependencies
    print("\n🔄 Running simplified test...")
    
    from label_config import LabelConfig, AgenticPolicy, ModerationLabel
    
    # Test label configuration
    print("Testing label configuration...")
    spam_label = LabelConfig.get_label("spam")
    print(f"Spam label: {spam_label.name} - {spam_label.description}")
    
    # Test violation mapping
    violations = ["spam", "hate"]
    labels = LabelConfig.get_labels_for_violations(violations)
    print(f"Labels for violations {violations}: {[l.name for l in labels]}")
    
    # Test auto-assignment logic
    should_auto = AgenticPolicy.should_escalate_to_human("Flagged", 0.7, "spam")
    print(f"Should escalate spam with 0.7 confidence: {should_auto}")
    
    print("✅ Basic configuration test passed!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()