"""Custom Ollama model wrapper for DeepEval using instructor library for JSON confinement."""

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI
from typing import Optional, Any
from pydantic import BaseModel
import instructor
import json


class CustomOllamaModel(DeepEvalBaseLLM):
    """Custom Ollama model that uses instructor library for robust JSON output."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434", temperature: float = 0.0):
        """Initialize the custom Ollama model with instructor.
        
        Args:
            model: The Ollama model name (e.g., "mistral:7b", "qwen3:8b")
            base_url: The Ollama server URL (will be converted to OpenAI-compatible endpoint)
            temperature: Temperature for generation (0.0 for deterministic output)
        """
        self.model = model
        self.temperature = temperature
        
        # Convert base URL to OpenAI-compatible endpoint
        # Ollama's OpenAI compatibility is at /v1
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
        
        # Create instructor-enhanced client with Ollama's OpenAI compatibility
        self.client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key="ollama",  # Required but unused for Ollama
            ),
            mode=instructor.Mode.JSON,  # Force JSON mode for structured outputs
        )
    
    def load_model(self):
        """Load the model (returns model name for DeepEval compatibility)."""
        return self.model
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Any:
        """Generate response with optional JSON schema enforcement using instructor.
        
        Args:
            prompt: The input prompt
            schema: Optional Pydantic model for JSON validation
            
        Returns:
            Either a string response or a validated Pydantic model instance
        """
        if schema:
            # Use instructor's response_model for structured output
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=schema,  # This enforces the schema
                    temperature=self.temperature,
                    max_retries=3,  # Instructor will retry if validation fails
                )
                return response
            except Exception as e:
                # Fallback: Create a minimal valid instance if all retries fail
                # This ensures evaluation continues even with problematic models
                print(f"Warning: Instructor failed after retries: {e}")
                try:
                    # Try to create a minimal valid instance
                    minimal_data = {}
                    if hasattr(schema, 'model_fields'):
                        # Pydantic v2
                        for field_name, field_info in schema.model_fields.items():
                            if field_info.is_required():
                                # Get the annotation type
                                field_type = field_info.annotation
                                # Provide minimal valid values
                                if field_type == str:
                                    minimal_data[field_name] = ""
                                elif field_type in (int, float):
                                    minimal_data[field_name] = 0
                                elif field_type == bool:
                                    minimal_data[field_name] = False
                                elif hasattr(field_type, '__origin__'):
                                    if field_type.__origin__ == list:
                                        minimal_data[field_name] = []
                                    elif field_type.__origin__ == dict:
                                        minimal_data[field_name] = {}
                    return schema(**minimal_data)
                except:
                    raise e
        else:
            # For non-structured output, use the underlying OpenAI client directly
            # (instructor requires response_model, so we bypass it for plain text)
            base_client = self.client.client  # Access the underlying OpenAI client
            response = base_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            # Extract the content from the response
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            return str(response)
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Any:
        """Async version of generate method.
        
        DeepEval uses async methods for evaluation, so we provide this wrapper.
        Note: Instructor's client is synchronous, so this is a sync-to-async wrapper.
        """
        # Run the sync version (similar to DeepEval's OllamaModel approach)
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        """Get the model name for DeepEval reporting."""
        return self.model