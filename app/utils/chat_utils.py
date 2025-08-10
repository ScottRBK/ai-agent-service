import re

def clean_response_for_memory(response: str) -> str:
    """Clean response before storing in memory"""
    response = re.sub(r'<think>.*?</think>|<thoughts>.*?</thoughts>', '', response, flags=re.DOTALL)
    response = response.replace('\\n', '\n')
    return response.strip()

def separate_chain_of_thought(response: str) -> tuple[str, str]:
    """Separate chain of thought from the main response
    
    Returns:
        tuple: (chain_of_thought_content, cleaned_response)
    """
    match = re.search(r'<think>(.*?)</think>|<thoughts>(.*?)</thoughts>', response, flags=re.DOTALL)
    if match:
        cot_content = (match.group(1) or match.group(2) or '').strip()
        cleaned_response = re.sub(r'<think>.*?</think>|<thoughts>.*?</thoughts>', '', response, flags=re.DOTALL).strip()
        return cot_content, cleaned_response
    return '', response.strip()