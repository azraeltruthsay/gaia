import re
import logging

logger = logging.getLogger("GAIA.OutputSanitizer")

def sanitize_llm_output(output: str) -> str:
    # Remove fabricated user messages
    pattern = re.compile(r"^#?\s*(User Message|You\s*>|User:)", re.IGNORECASE | re.MULTILINE)
    if pattern.search(output):
        logger.warning("⛔ Fabricated user message detected in LLM response. Removing.")
        output = pattern.sub("[REMOVED FABRICATION]", output)

    # Limit excessive backticks
    max_backticks = 10
    if output.count("```") > max_backticks:
        parts = output.split("```")
        output = "```".join(parts[:max_backticks]) + "\n[...truncated excessive output...]"

    return output


def enforce_single_response(output: str) -> str:
    # Remove any multi-turn Q&A blocks, like "User Prompt", "Response", etc.
    patterns = [
        r"(?i)(#?\s*(User Prompt|User Message|Assistant Response|Response):).*",
        r"(?i)(\*\*User Input\*\*|Prompt|Response|Task Update|Action Suggestion|Clarification Needed|Mistake Correction|Resource Recommendations|Code Integrity):.*",
    ]
    for pattern in patterns:
        output = re.sub(pattern, "", output)

    # Optionally: Cut output after first answer
    if "# GAIA's Response:" in output:
        output = output.split("# GAIA's Response:", 1)[-1].strip()

    return output.strip()

def enforce_file_verification(output: str, user_prompt: str) -> str:
    # Check if the user is asking about a file
    file_patterns = [r"\.json\b", r"\.md\b", r"\.py\b", r"\.log\b", r"read the file", r"what is in the file"]
    if any(re.search(pattern, user_prompt.lower()) for pattern in file_patterns):
        # Check if an EXECUTE directive or ai.read() is present in output
        if not re.search(r"(EXECUTE:\s*ai\.read|ai\.read\()", output):
            logger.warning("⛔ File-related response detected without file access. Blocking fabrication.")
            return "⚠️ File content must be verified via primitives. Please include an EXECUTE: ai.read() directive or request the correct primitive."
    
    return output