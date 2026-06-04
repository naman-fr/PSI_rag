import pytest
from app.cache.redis_client import InMemoryCache
from app.memory.conversation import ConversationManager
from app.memory.summary import SummaryManager


@pytest.mark.asyncio
async def test_conversation_manager():
    cache = InMemoryCache()
    manager = ConversationManager(cache)
    
    username = "user1"
    session_id = "sess1"
    
    await manager.add_message(username, session_id, "user", "Hello there")
    await manager.add_message(username, session_id, "assistant", "Hi! How can I help?")
    
    history = await manager.get_full_history(username, session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello there"
    
    recent = await manager.get_recent_messages(username, session_id, limit=1)
    assert len(recent) == 1
    assert recent[0]["role"] == "assistant"
    
    await manager.clear_conversation(username, session_id)
    history_after = await manager.get_full_history(username, session_id)
    assert len(history_after) == 0


@pytest.mark.asyncio
async def test_summary_manager():
    cache = InMemoryCache()
    manager = SummaryManager(cache)
    
    username = "user1"
    assert await manager.get_summary(username) is None
    
    await manager.update_summary(username, "Summary text")
    assert await manager.get_summary(username) == "Summary text"
    
    assert SummaryManager.should_summarize(10, interval=5) is True
    assert SummaryManager.should_summarize(3, interval=5) is False
