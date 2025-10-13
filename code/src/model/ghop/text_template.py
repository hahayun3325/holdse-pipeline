# code/src/model/ghop/text_template.py
"""
Text template generation for GHOP conditioning.
Official Obj2Text implementation from GHOP project (utils/obj2text.py).

This maps object category names to natural language prompts for text-conditioned
diffusion guidance.
"""

import json
import os


class Obj2Text:
    """
    Maps object categories to text prompts for GHOP text conditioning.

    Official implementation from GHOP project.
    Converts category names (e.g., "mug") to prompts (e.g., "an image of a hand grasping a mug").

    The text library (JSON file) provides optional category name normalization:
    - Maps raw category names to canonical names
    - Example: {"coffee_mug": "mug", "water_bottle": "bottle"}

    Args:
        lib: Optional path to JSON file with category name mappings

    Example:
        >>> obj2text = Obj2Text()
        >>> prompt = obj2text("mug")
        >>> print(prompt)
        'an image of a hand grasping a mug'

        >>> prompts = obj2text(["mug", "bottle"])
        >>> print(prompts)
        ['an image of a hand grasping a mug', 'an image of a hand grasping a bottle']
    """

    def __init__(self, lib=None):
        """
        Initialize text template mapper.

        Args:
            lib: Optional path to JSON file with category name mappings.
                 Format: {"raw_name": "canonical_name", ...}
                 Example: {"coffee_mug": "mug", "plastic_bottle": "bottle"}
        """
        if lib is None:
            self.lib = None
            self.lower_lib = None
        else:
            print("loading word mapping: ", lib)
            self.lib = json.load(open(lib))
            # Create lowercase version for case-insensitive lookup
            self.lower_lib = {k.lower(): v for k, v in self.lib.items()}

        # Official GHOP template
        self.template = "an image of a hand grasping a {}"

    def __call__(self, text):
        """
        Generate text prompt(s) from object category name(s).

        Args:
            text: str or List[str] - Object category name(s)

        Returns:
            prompt: str or List[str] - Natural language prompt(s)

        Behavior:
            - If text is already a prompt (starts with template prefix), returns as-is
            - If library is provided, looks up canonical name
            - Otherwise, directly inserts text into template

        Examples:
            >>> obj2text = Obj2Text()
            >>> obj2text("mug")
            'an image of a hand grasping a mug'

            >>> obj2text(["bottle", "cup"])
            ['an image of a hand grasping a bottle', 'an image of a hand grasping a cup']

            >>> obj2text("")  # Empty string
            ''

            >>> obj2text("an image of a hand grasping a phone")  # Already formatted
            'an image of a hand grasping a phone'
        """
        if isinstance(text, str):
            # Handle empty string
            if len(text) == 0:
                return ""

            # Already formatted - return as-is
            if text.startswith("an image of a hand grasping"):
                # print('emmm nested text???')
                return text

            # No library - use raw text
            if self.lib is None:
                return self.template.format(text)

            # Library provided - lookup canonical name
            else:
                try:
                    # Try lowercase lookup first
                    text = self.lower_lib[text.lower()]
                except KeyError:
                    try:
                        # Try exact case match
                        text = self.lib[text]
                    except KeyError:
                        # Not found - use raw text with warning
                        print(f"cannot find {text} in lib???")

                return self.template.format(text)

        elif isinstance(text, list):
            # Recursively process list
            return [self(t) for t in text]


# ============================================================================
# Convenience Factory Function
# ============================================================================

def create_text_template(lib_name=None):
    """
    Factory function to create Obj2Text instance.

    Args:
        lib_name: Optional path to JSON file with category mappings

    Returns:
        Obj2Text instance

    Example:
        >>> obj2text = create_text_template("data/ghop/text_templates.json")
        >>> prompt = obj2text("mug")
    """
    return Obj2Text(lib=lib_name)


# ============================================================================
# Example Library Format
# ============================================================================

EXAMPLE_LIBRARY_FORMAT = """
{
  "coffee_mug": "mug",
  "tea_cup": "cup",
  "water_bottle": "bottle",
  "plastic_bottle": "bottle",
  "smartphone": "phone",
  "cellphone": "phone",
  "screwdriver": "tool",
  "wrench": "tool"
}

This maps various category names to canonical names that work well
with the GHOP diffusion model.
"""


# ============================================================================
# Unit Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Obj2Text (GHOP official implementation)...")
    print("=" * 70)

    # Test 1: No library (default)
    print("\n[Test 1] No library - direct text insertion")
    obj2text = Obj2Text()

    result = obj2text("mug")
    expected = "an image of a hand grasping a mug"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"✓ Single category: '{result}'")

    # Test 2: List of categories
    print("\n[Test 2] List of categories")
    results = obj2text(["mug", "bottle", "phone"])
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print(f"✓ Batch processing: {len(results)} prompts")
    for cat, prompt in zip(["mug", "bottle", "phone"], results):
        print(f"  - {cat}: {prompt}")

    # Test 3: Empty string
    print("\n[Test 3] Empty string handling")
    result = obj2text("")
    assert result == "", f"Expected empty string, got '{result}'"
    print(f"✓ Empty string returns: '{result}'")

    # Test 4: Already formatted text
    print("\n[Test 4] Already formatted text")
    already_formatted = "an image of a hand grasping a hammer"
    result = obj2text(already_formatted)
    assert result == already_formatted, f"Should return as-is"
    print(f"✓ Already formatted: '{result}'")

    # Test 5: With library (using temporary file)
    print("\n[Test 5] With category mapping library")
    import tempfile
    import os

    # Create temporary library
    lib_data = {
        "coffee_mug": "mug",
        "plastic_bottle": "bottle",
        "PHONE": "phone"  # Test case insensitivity
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(lib_data, f)
        temp_lib = f.name

    try:
        obj2text_lib = Obj2Text(lib=temp_lib)

        # Test exact match
        result = obj2text_lib("coffee_mug")
        expected = "an image of a hand grasping a mug"
        assert result == expected, f"Library lookup failed"
        print(f"✓ Library lookup (exact): 'coffee_mug' -> '{result}'")

        # Test case-insensitive match
        result = obj2text_lib("phone")  # lowercase lookup for "PHONE"
        expected = "an image of a hand grasping a phone"
        assert result == expected, f"Case-insensitive lookup failed"
        print(f"✓ Library lookup (case-insensitive): 'phone' -> '{result}'")

        # Test missing key (should still work with warning)
        result = obj2text_lib("unknown_category")
        expected = "an image of a hand grasping a unknown_category"
        assert result == expected, f"Fallback failed"
        print(f"✓ Fallback for unknown: 'unknown_category' -> '{result}'")

    finally:
        os.remove(temp_lib)

    # Test 6: Factory function
    print("\n[Test 6] Factory function")
    obj2text_factory = create_text_template()
    result = obj2text_factory("bowl")
    expected = "an image of a hand grasping a bowl"
    assert result == expected
    print(f"✓ Factory function works: '{result}'")

    print("\n" + "=" * 70)
    print("✓ All tests passed! Obj2Text is ready for Phase 2.")
    print("\nUsage:")
    print("  from src.model.ghop.text_template import Obj2Text")
    print("  obj2text = Obj2Text()")
    print("  prompt = obj2text('mug')")