import json
import re
from loguru import logger


def clean_html_response(html_content, previous_content=None, max_retries=3):
    """
    Clean the HTML response from Gemini.
    
    Args:
        html_content (str): The HTML content to clean
        previous_content (str, optional): Previous partial HTML content to continue from
        max_retries (int, optional): Maximum number of retries for partial content
        
    Returns:
        str: The cleaned HTML content
    """
    if html_content is None:
        return None
    
    # Check if the new content itself contains a complete HTML document
    # This handles the case where continuation generates a full HTML instead of just the missing part
    if previous_content:
        # Check for complete HTML in the new content
        html_match_in_new = re.search(
            r"(?:<\?xml.*?\?>)?(?:<\!DOCTYPE.*?>)?(?:<html.*?>).*?<\/html>", html_content, re.DOTALL
        )
        
        if html_match_in_new:
            logger.info("Detected complete HTML in continuation response, using it directly")
            # Use only the new content since it's complete
            # No need to combine with previous_content
        else:
            # Continue with normal continuation by combining previous and new content
            logger.info("Continuing from previous partial HTML content")
            html_content = previous_content + html_content
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # First try to parse as JSON in case response was JSON-encoded
            try:
                parsed = json.loads(html_content)
                if isinstance(parsed, str):
                    html_content = parsed
                elif isinstance(parsed, dict) and "html" in parsed:
                    html_content = parsed["html"]
            except json.JSONDecodeError:
                pass  # Not JSON, proceed with normal cleaning
                
            # Clean up markdown code blocks
            html_content = re.sub(r"```html\s*", "", html_content)
            html_content = re.sub(r"```\s*$", "", html_content)
            html_content = re.sub(r"```[a-zA-Z]*\s*", "", html_content)
            
            # Extract HTML content
            # Try to match full HTML document first - support both HTML and XML declarations
            html_match = re.search(
                r"(?:<\?xml.*?\?>)?(?:<\!DOCTYPE.*?>)?(?:<html.*?>).*?<\/html>", html_content, re.DOTALL
            )
            
            # If no match, try to match just the body content
            if not html_match:
                html_match = re.search(
                    r"(?:<body.*?>).*?<\/body>", html_content, re.DOTALL
                )
                
            # If still no match, try to match any HTML-like content between div tags
            if not html_match:
                html_match = re.search(
                    r"<div.*?>.*?<\/div>", html_content, re.DOTALL
                )
                
            if not html_match:
                # Check if we have a partial HTML document (opening tags without closing tags)
                # This indicates we need to get more content
                has_opening_html = re.search(r"<html.*?>", html_content, re.DOTALL) is not None
                has_opening_body = re.search(r"<body.*?>", html_content, re.DOTALL) is not None
                has_opening_div = re.search(r"<div.*?>", html_content, re.DOTALL) is not None
                
                if has_opening_html or has_opening_body or has_opening_div:
                    # We have a partial document, return it for continuation
                    logger.warning("Detected partial HTML document, returning for continuation")
                    return {
                        "status": "partial",
                        "content": html_content
                    }
                
                # If we can't find valid HTML, log the raw response for debugging
                logger.error("Failed to find valid HTML structure in response")
                logger.debug("Response content structure:")
                logger.debug("-" * 40)
                # Remove extra newlines for better readability
                html_content = re.sub(r"\n{2,}", "\n", html_content)
                logger.debug(f"{html_content[:500]}\n ... \n{html_content[-500:]}")
                logger.debug("-" * 40)
                
                if retry_count < max_retries - 1:
                    logger.warning(f"Retry {retry_count + 1}/{max_retries} for HTML cleaning")
                    retry_count += 1
                    continue
                else:
                    raise ValueError("Could not find valid HTML content in response")
                
            if html_match:
                html_content = html_match.group(0)
                
            # Final cleanup
            html_content = html_content.strip()
            if not html_content:
                if retry_count < max_retries - 1:
                    logger.warning(f"Retry {retry_count + 1}/{max_retries} for empty HTML content")
                    retry_count += 1
                    continue
                else:
                    raise ValueError("Cleaned HTML content is empty")
                
            return html_content
            
        except Exception as e:
            if retry_count < max_retries - 1:
                logger.warning(f"Retry {retry_count + 1}/{max_retries} after error: {str(e)}")
                retry_count += 1
                continue
            else:
                logger.error(f"Error cleaning HTML response after {max_retries} retries: {str(e)}")
                raise ValueError("Failed to clean HTML response") from e
