"""
Changelog reading utilities for GPE Lab.

Provides functions to read and parse the changelog file.
"""
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Path to changelog file relative to project root
CHANGELOG_FILENAME = "changelog_zh.md"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_changelog_path() -> Path:
    """
    Get the path to the changelog file.
    
    Returns:
        Path to data/changelog_zh.md
    """
    return get_project_root() / "data" / CHANGELOG_FILENAME


def read_changelog_markdown(path: Optional[Path] = None) -> tuple[bool, str]:
    """
    Read the changelog markdown file.
    
    Args:
        path: Optional custom path to changelog file. 
              Defaults to data/changelog_zh.md
              
    Returns:
        Tuple of (success: bool, content: str)
        - If success=True, content is the markdown text
        - If success=False, content is an error/guidance message
    """
    if path is None:
        path = get_changelog_path()
    
    if not path.exists():
        logger.warning(f"Changelog file not found: {path}")
        return False, ""
    
    try:
        content = path.read_text(encoding="utf-8")
        return True, content
    except Exception as e:
        logger.error(f"Failed to read changelog: {e}")
        return False, ""


def filter_changelog_content(content: str, search_term: str) -> str:
    """
    Filter changelog content by search term.
    
    Simple implementation: returns sections (## headers) that contain
    the search term in their content.
    
    Args:
        content: Full changelog markdown
        search_term: Term to search for (case-insensitive)
        
    Returns:
        Filtered markdown content, or original if search_term is empty
    """
    if not search_term.strip():
        return content
    
    search_lower = search_term.lower()
    lines = content.split("\n")
    
    result_lines = []
    current_section = []
    section_matches = False
    in_header = False
    
    for line in lines:
        # Check if this is a level-2 header (## ...)
        if line.startswith("## "):
            # Save previous section if it matched
            if current_section and section_matches:
                result_lines.extend(current_section)
                result_lines.append("")  # Add spacing
            
            # Start new section
            current_section = [line]
            section_matches = search_lower in line.lower()
            in_header = True
        elif line.startswith("# ") and not line.startswith("## "):
            # Top-level header, always include
            if current_section and section_matches:
                result_lines.extend(current_section)
                result_lines.append("")
            
            result_lines.append(line)
            current_section = []
            section_matches = False
        else:
            current_section.append(line)
            if search_lower in line.lower():
                section_matches = True
    
    # Don't forget last section
    if current_section and section_matches:
        result_lines.extend(current_section)
    
    if not result_lines:
        return f"*没有找到包含 \"{search_term}\" 的更新记录*"
    
    return "\n".join(result_lines)
