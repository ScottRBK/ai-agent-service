import re

def clean_response_for_memory(response: str) -> str:
    """Clean response before storing in memory"""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.replace('\\n', '\n')
    return response.strip() 