"""
UPSC-optimized prompts for relationship classification in atomic notes.

Relationship Types:
- PREREQUISITE: Must understand this first
- RELATED: Same domain, complementary information  
- COMPARISON: Contrasting concepts (India vs USA, etc.)
- CAUSE_EFFECT: Historical/logical causation
- EXAMPLE: Practical application or case study
- PART_OF: Component relationship (Article within Part)
"""

from enum import Enum
from typing import List

class LinkType(Enum):
    PREREQUISITE = "prerequisite"
    RELATED = "related"
    COMPARISON = "comparison"
    CAUSE_EFFECT = "cause_effect"
    EXAMPLE = "example"
    PART_OF = "part_of"
    UNRELATED = "unrelated"

    @property
    def display_name(self) -> str:
        """Human-readable name for display in markdown."""
        mapping = {
            "prerequisite": "Prerequisites",
            "related": "Related Concepts",
            "comparison": "Comparisons",
            "cause_effect": "Cause & Effect",
            "example": "Examples & Applications",
            "part_of": "Part Of",
            "unrelated": None
        }
        return mapping.get(self.value)


CLASSIFY_RELATIONSHIP_PROMPT = """You are an expert UPSC exam tutor analyzing relationships between study notes.

Given two notes, classify the relationship between them. The FIRST note is the SOURCE (the note being updated), and the SECOND note is the TARGET (the note being linked to).

Relationship Types:
1. **PREREQUISITE**: TARGET is foundational knowledge needed to understand SOURCE
   - Example: "Federal Structure" is prerequisite for "Centre-State Relations"
   
2. **RELATED**: Same topic domain, complementary but distinct information
   - Example: "Fundamental Rights" and "Fundamental Duties" are related
   
3. **COMPARISON**: Contrasting concepts, often India vs other countries
   - Example: "Indian Single Citizenship" compares to "US Dual Citizenship"
   
4. **CAUSE_EFFECT**: Historical or logical causation chain
   - Example: "Bengal Partition 1905" caused "Swadeshi Movement"
   
5. **EXAMPLE**: TARGET is a practical application, case study, or instance of SOURCE
   - Example: "Kesavananda Bharati Case" is an example for "Basic Structure Doctrine"
   
6. **PART_OF**: TARGET is a component/article within SOURCE (or vice versa)
   - Example: "Article 21" is part of "Fundamental Rights"

7. **UNRELATED**: Notes have no meaningful academic connection

---

SOURCE NOTE:
{source_content}

TARGET NOTE:  
{target_content}

---

Analyze carefully and respond with ONLY the relationship type in uppercase: PREREQUISITE, RELATED, COMPARISON, CAUSE_EFFECT, EXAMPLE, PART_OF, or UNRELATED

Relationship:"""


BATCH_CLASSIFY_PROMPT = """You are an expert UPSC exam tutor analyzing relationships between study notes.

Given a SOURCE note and multiple TARGET notes, classify each relationship.

Relationship Types:
- PREREQUISITE: Target is foundational knowledge for source
- RELATED: Same domain, complementary information
- COMPARISON: Contrasting concepts (India vs USA, etc.)
- CAUSE_EFFECT: Historical/logical causation
- EXAMPLE: Target is case study/application of source
- PART_OF: Component relationship
- UNRELATED: No meaningful connection

---

SOURCE NOTE:
{source_content}

---

TARGET NOTES:
{targets_content}

---

For each target note, respond with its number and relationship type, one per line.
Format: [number]: [RELATIONSHIP_TYPE]

Example output:
1: RELATED
2: COMPARISON
3: UNRELATED
4: PREREQUISITE

Classifications:"""


def format_targets_for_batch(targets: List[tuple]) -> str:
    """Format multiple target notes for batch classification prompt."""
    formatted = []
    for i, (name, content_preview) in enumerate(targets, 1):
        # Truncate content for prompt efficiency
        preview = content_preview[:300] + "..." if len(content_preview) > 300 else content_preview
        formatted.append(f"[{i}] **{name}**\n{preview}\n")
    return "\n".join(formatted)


def parse_batch_response(response: str, num_targets: int) -> List[LinkType]:
    """Parse LLM batch response into list of LinkTypes."""
    import re
    
    results = [LinkType.UNRELATED] * num_targets  # Default
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Match patterns like "1: RELATED" or "1. PREREQUISITE" or just "1 RELATED"
        match = re.match(r'^\[?(\d+)\]?[:\.\s]+(\w+)', line)
        if match:
            idx = int(match.group(1)) - 1  # Convert to 0-indexed
            type_str = match.group(2).upper()
            
            if 0 <= idx < num_targets:
                try:
                    results[idx] = LinkType(type_str.lower())
                except ValueError:
                    # Unknown type, keep as UNRELATED
                    pass
    
    return results
