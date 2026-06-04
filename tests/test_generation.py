import pytest
from unittest.mock import MagicMock, patch
from app.rag.generation import LLMService


@pytest.mark.asyncio
async def test_llm_service_generate():
    # Patch the Groq client in app.rag.generation
    with patch("app.rag.generation.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        # Setup mock response
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Mocked answer"
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = LLMService()
        messages = [{"role": "user", "content": "Hello"}]
        text, usage = await service.generate(messages, max_tokens=10, temperature=0.0)
        
        assert text == "Mocked answer"
        assert usage["total_tokens"] == 15
