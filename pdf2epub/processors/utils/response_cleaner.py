"""
Utilities for cleaning LLM responses.

These are pure functions for cleaning markdown responses from LLMs.
"""


def clean_markdown_response(content: str) -> str:
    """
    Clean up markdown response from LLM.

    Removes code block wrappers that LLMs sometimes add.

    Args:
        content: Raw response from LLM

    Returns:
        Cleaned markdown content
    """
    lines = content.strip().split('\n')

    # Look for code block markers in first 3 non-empty lines
    non_empty_count = 0
    code_block_start = -1

    for i, line in enumerate(lines):
        if line.strip():  # Non-empty line
            non_empty_count += 1
            # Check if this line is a code block marker
            if line.strip() in ['```markdown', '```'] or line.strip().startswith('```'):
                code_block_start = i + 1  # Start from the line after the marker
                break
            if non_empty_count >= 3:
                break

    # If we found a code block marker, remove everything before and including it
    if code_block_start > 0:
        lines = lines[code_block_start:]

    # Rejoin the content
    content = '\n'.join(lines)

    # Also handle case where ``` appears at the end
    if content.strip().endswith('```'):
        lines = content.strip().split('\n')
        if lines[-1].strip() == '```':
            lines = lines[:-1]
            content = '\n'.join(lines)

    return content.strip()
