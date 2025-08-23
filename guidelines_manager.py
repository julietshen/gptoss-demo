"""
Guidelines Management System
Allows users to upload, edit, and manage custom moderation guidelines
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class GuidelineSet:
    """Represents a set of moderation guidelines."""
    name: str
    description: str
    categories: List[Dict[str, str]]  # [{"name": "category", "description": "desc", "examples": ["ex1", "ex2"]}]
    decision_principles: List[str]
    draft_response_template: str
    created_at: datetime
    last_modified: datetime
    active: bool = False
    custom_prompt: Optional[str] = None

class GuidelinesManager:
    """Manages multiple sets of moderation guidelines."""
    
    # Default anti-misogyny & anti-transphobia guidelines
    DEFAULT_GUIDELINES = GuidelineSet(
        name="Anti-Misogyny & Anti-Transphobia",
        description="Specialized guidelines for detecting misogynistic and transphobic content",
        categories=[
            {
                "name": "Misogyny - Explicit",
                "description": "Direct slurs, statements of female inferiority, calls for violence against women",
                "examples": [
                    "Direct slurs targeting women (b*tch, c*nt, wh*re used as attacks)",
                    "Explicit statements that women are inferior or should be subordinated",
                    "Calls for violence against women or feminine individuals"
                ]
            },
            {
                "name": "Misogyny - Subtle",
                "description": "Dismissive language, capability generalizations, victim-blaming",
                "examples": [
                    "Generalizations about women's capabilities (\"women can't drive/do math\")",
                    "Dismissive language about women's concerns (\"Karen,\" \"emotional,\" \"hysterical\")",
                    "Objectifying comments focusing solely on physical appearance in professional contexts"
                ]
            },
            {
                "name": "Misogyny - Harassment",
                "description": "Targeted gendered attacks, coordinated harassment, threats of sexual violence",
                "examples": [
                    "Coordinated attacks on women's accounts",
                    "Gendered insults in response to women's opinions",
                    "Threats of sexual violence"
                ]
            },
            {
                "name": "Transphobia - Explicit",
                "description": "Deliberate misgendering, slurs, denying trans existence, calls for violence",
                "examples": [
                    "Deliberate misgendering with hostile intent",
                    "Slurs targeting transgender individuals (tr*nny, etc.)",
                    "Statements denying the existence or validity of transgender people"
                ]
            },
            {
                "name": "Transphobia - Subtle",
                "description": "\"Biological sex\" exclusion arguments, \"real women/men\" language, bathroom/sports fearmongering",
                "examples": [
                    "\"Biological sex\" arguments used to exclude trans people",
                    "Concern trolling about \"protecting children\" from trans people",
                    "References to \"real women/men\" that exclude trans people"
                ]
            },
            {
                "name": "Transphobia - Deadnaming",
                "description": "Using former names/identities of trans people",
                "examples": [
                    "Using a trans person's birth name after being corrected",
                    "Sharing previous names/identities of trans individuals",
                    "Historical references designed to invalidate current identity"
                ]
            },
            {
                "name": "Transphobia - Medicalization",
                "description": "Calling being trans a \"mental illness\", unsolicited medical advice, detransition promotion",
                "examples": [
                    "Referring to being transgender as a \"mental illness\" or \"disorder\"",
                    "Unsolicited medical advice or demands for medical information",
                    "Discussions of \"detransition\" intended to discourage transition"
                ]
            }
        ],
        decision_principles=[
            "Focus on impact over intent",
            "Consider intersectionality (racism + misogyny, etc.)",
            "Distinguish education from perpetuation",
            "Account for reclaimed language by affected communities"
        ],
        draft_response_template="pls dont post this tyvm",
        created_at=datetime.now(),
        last_modified=datetime.now(),
        active=True
    )
    
    def __init__(self, guidelines_file: str = "custom_guidelines.json"):
        """
        Initialize the guidelines manager.
        
        Args:
            guidelines_file: File to store custom guidelines
        """
        self.guidelines_file = guidelines_file
        self.guidelines = {}
        self.active_guidelines = None
        self.load_guidelines()
    
    def load_guidelines(self):
        """Load guidelines from file or set defaults."""
        try:
            if os.path.exists(self.guidelines_file):
                with open(self.guidelines_file, 'r') as f:
                    data = json.load(f)
                    for name, guideline_data in data.items():
                        # Convert datetime strings back to datetime objects
                        guideline_data['created_at'] = datetime.fromisoformat(guideline_data['created_at'])
                        guideline_data['last_modified'] = datetime.fromisoformat(guideline_data['last_modified'])
                        self.guidelines[name] = GuidelineSet(**guideline_data)
                        
                        if guideline_data.get('active', False):
                            self.active_guidelines = self.guidelines[name]
                            
                logger.info(f"Loaded {len(self.guidelines)} guideline sets")
            else:
                logger.info("No custom guidelines file found, using defaults")
        except Exception as e:
            logger.error(f"Error loading guidelines: {e}")
        
        # Set default if no active guidelines
        if not self.active_guidelines:
            self.guidelines["default"] = self.DEFAULT_GUIDELINES
            self.active_guidelines = self.DEFAULT_GUIDELINES
    
    def save_guidelines(self):
        """Save guidelines to file."""
        try:
            # Convert guidelines to dict format for JSON serialization
            data = {}
            for name, guideline_set in self.guidelines.items():
                guideline_data = asdict(guideline_set)
                guideline_data['created_at'] = guideline_set.created_at.isoformat()
                guideline_data['last_modified'] = guideline_set.last_modified.isoformat()
                data[name] = guideline_data
            
            with open(self.guidelines_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.guidelines)} guideline sets")
        except Exception as e:
            logger.error(f"Error saving guidelines: {e}")
    
    def add_guidelines(self, guideline_set: GuidelineSet) -> bool:
        """
        Add new guidelines.
        
        Args:
            guideline_set: Guidelines to add
            
        Returns:
            True if successful
        """
        try:
            self.guidelines[guideline_set.name] = guideline_set
            self.save_guidelines()
            logger.info(f"Added guidelines: {guideline_set.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding guidelines: {e}")
            return False
    
    def set_active_guidelines(self, name: str) -> bool:
        """
        Set which guidelines are currently active.
        
        Args:
            name: Name of guidelines to activate
            
        Returns:
            True if successful
        """
        if name in self.guidelines:
            # Deactivate all guidelines
            for guideline_set in self.guidelines.values():
                guideline_set.active = False
            
            # Activate selected guidelines
            self.guidelines[name].active = True
            self.active_guidelines = self.guidelines[name]
            self.save_guidelines()
            logger.info(f"Activated guidelines: {name}")
            return True
        else:
            logger.error(f"Guidelines not found: {name}")
            return False
    
    def get_active_guidelines(self) -> GuidelineSet:
        """Get currently active guidelines."""
        return self.active_guidelines or self.DEFAULT_GUIDELINES
    
    def get_all_guidelines(self) -> Dict[str, GuidelineSet]:
        """Get all available guidelines."""
        return self.guidelines
    
    def delete_guidelines(self, name: str) -> bool:
        """
        Delete guidelines.
        
        Args:
            name: Name of guidelines to delete
            
        Returns:
            True if successful
        """
        if name in self.guidelines and name != "default":
            # If deleting active guidelines, switch to default
            if self.active_guidelines and self.active_guidelines.name == name:
                self.set_active_guidelines("default")
            
            del self.guidelines[name]
            self.save_guidelines()
            logger.info(f"Deleted guidelines: {name}")
            return True
        else:
            logger.error(f"Cannot delete guidelines: {name}")
            return False
    
    def generate_prompt_section(self) -> str:
        """Generate the guidelines section for the moderation prompt."""
        guidelines = self.get_active_guidelines()
        
        # Build categories section
        categories_text = ""
        for category in guidelines.categories:
            categories_text += f"\n**{category['name']}:**\n"
            categories_text += f"- {category['description']}\n"
            if category.get('examples'):
                categories_text += "- Examples:\n"
                for example in category['examples'][:3]:  # Limit to 3 examples
                    categories_text += f"  • {example}\n"
        
        # Build principles section
        principles_text = ""
        for principle in guidelines.decision_principles:
            principles_text += f"- {principle}\n"
        
        return f"""**Moderation Guidelines: {guidelines.name}**

**Key Categories to Detect:**
{categories_text}

**Decision Principles:**
{principles_text}"""
    
    def get_draft_response_template(self) -> str:
        """Get the draft response template from active guidelines."""
        guidelines = self.get_active_guidelines()
        return guidelines.draft_response_template
    
    def import_from_markdown(self, markdown_text: str, name: str, description: str = "") -> GuidelineSet:
        """
        Import guidelines from markdown format.
        
        Args:
            markdown_text: Markdown content with guidelines
            name: Name for the guideline set
            description: Description of the guidelines
            
        Returns:
            Created GuidelineSet
        """
        categories = []
        decision_principles = []
        draft_response = "pls dont post this tyvm"  # Default
        
        lines = markdown_text.split('\n')
        current_category = None
        current_examples = []
        in_principles = False
        
        for line in lines:
            line = line.strip()
            
            # Look for category headers (## or ###)
            if line.startswith('##') and not line.startswith('###'):
                # Save previous category
                if current_category:
                    current_category['examples'] = current_examples
                    categories.append(current_category)
                
                # Start new category
                category_name = line.replace('#', '').strip()
                current_category = {
                    'name': category_name,
                    'description': '',
                    'examples': []
                }
                current_examples = []
                in_principles = False
            
            # Look for principles section
            elif 'principle' in line.lower() or 'guideline' in line.lower():
                in_principles = True
                if current_category:
                    current_category['examples'] = current_examples
                    categories.append(current_category)
                    current_category = None
            
            # Extract description (first paragraph after category)
            elif current_category and not current_category['description'] and line and not line.startswith('-'):
                current_category['description'] = line
            
            # Extract examples and principles
            elif line.startswith('-') or line.startswith('•'):
                content = line[1:].strip()
                if in_principles:
                    decision_principles.append(content)
                elif current_category:
                    current_examples.append(content)
            
            # Look for draft response template
            elif 'response' in line.lower() and ':' in line:
                draft_response = line.split(':', 1)[1].strip().strip('"\'')
        
        # Save last category
        if current_category:
            current_category['examples'] = current_examples
            categories.append(current_category)
        
        return GuidelineSet(
            name=name,
            description=description,
            categories=categories,
            decision_principles=decision_principles,
            draft_response_template=draft_response,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            active=False
        )