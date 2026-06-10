"""
Main RAG pipeline orchestrator.

Coordinates the full flow: input validation -> intent classification ->
retrieval -> generation -> verification -> caching -> response.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from langsmith import traceable

from app.core.config import get_settings
from app.core.constants import REFUSAL_RESPONSE
from app.core.logging import get_logger

logger = get_logger("orchestrator")


class RAGOrchestrator:
    """Orchestrates the full RAG pipeline for a single request."""

    def __init__(
        self,
        embedding_service,
        retriever,
        llm_service,
        cache_backend,
        conversation_manager,
        summary_manager,
        metrics_collector=None,
        experiment_tracker=None,
    ):
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.llm_service = llm_service
        self.cache = cache_backend
        self.conversation = conversation_manager
        self.summary = summary_manager
        self.metrics = metrics_collector
        self.tracker = experiment_tracker
        self.settings = get_settings()

    @traceable(name="answer_question")
    async def process_query(
        self,
        question: str,
        username: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the full RAG pipeline.

        Flow:
        1. Generate trace ID
        2. Input sanitization + injection detection
        3. Intent classification (greeting vs rag vs adversarial)
        4. Check response cache
        5. Embed query + retrieve context
        6. Retrieval quality gate
        7. Assemble context with memory
        8. Generate grounded answer
        9. Verify answer (hallucination check)
        10. Apply refusal policy
        11. Cache response + store conversation
        12. Return structured response
        """
        import time

        start_time = time.time()
        trace_id = str(uuid.uuid4())[:12]
        session_id = session_id or f"session_{username}"
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "processing_query",
            trace_id=trace_id,
            username=username,
            question_length=len(question),
        )

        # --- Step 1: Input Guard ---
        from app.guardrails.input_guard import detect_injection, sanitize_input

        question = sanitize_input(question)

        if detect_injection(question):
            logger.warning("adversarial_input_detected", trace_id=trace_id)
            return self._build_response(
                trace_id=trace_id,
                mode="guardrail_refusal",
                answer="I can only answer questions about the provided documents. "
                "Please ask a relevant question.",
                confidence=0.0,
                timestamp=timestamp,
            )

        # --- Step 2: Intent Classification ---
        from app.guardrails.intent_classifier import classify_intent, direct_chat_reply

        intent = classify_intent(question)

        if intent == "adversarial":
            logger.warning("adversarial_intent", trace_id=trace_id)
            return self._build_response(
                trace_id=trace_id,
                mode="guardrail_refusal",
                answer="I can only answer questions about the provided documents.",
                confidence=0.0,
                timestamp=timestamp,
            )

        if intent == "greeting":
            reply = direct_chat_reply(question)
            return self._build_response(
                trace_id=trace_id,
                mode="direct",
                answer=reply,
                confidence=1.0,
                verdict={"supported": True, "confidence": 1.0, "reason": "direct_smalltalk"},
                timestamp=timestamp,
            )

        # --- Step 3: Check Response Cache ---
        from app.cache.response_cache import get_cached_response, make_cache_key

        cache_key = make_cache_key(question, self.settings.groq_model)
        cached = await get_cached_response(self.cache, cache_key)
        if cached:
            logger.info("cache_hit", trace_id=trace_id)
            cached["trace_id"] = trace_id
            cached["cached"] = True
            cached["mode"] = "cached"
            if self.metrics:
                latency = time.time() - start_time
                self.metrics.record_request("cached", latency, 0, True)
            return cached

        # --- Step 4: Retrieve Context ---
        import numpy as np

        query_vector = self.embedding_service.embed_query(question)

        # Retrieve a larger candidate pool to allow cross-document balancing in reranking
        candidate_k = max(12, self.settings.top_k * 2)
        results = self.retriever.search(
            query_vector=query_vector,
            top_k=candidate_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )

        # --- Step 5: Retrieval Gate ---
        from app.guardrails.retrieval_gate import check_retrieval_quality

        gate_passed, gate_reason = check_retrieval_quality(
            results, min_score=self.settings.retrieval_score_threshold, min_chunks=1
        )

        if not gate_passed:
            logger.info("retrieval_gate_failed", trace_id=trace_id, reason=gate_reason)
            return self._build_response(
                trace_id=trace_id,
                mode="guardrail_refusal",
                answer=REFUSAL_RESPONSE,
                confidence=0.0,
                verdict={"supported": False, "confidence": 0.0, "reason": gate_reason},
                timestamp=timestamp,
            )

        # --- Step 6: Rerank & Deduplicate ---
        from app.rag.reranking import rerank_results

        results = rerank_results(results, top_k=self.settings.top_k)

        # --- Step 7: Assemble Context with Memory ---
        from app.rag.context import assemble_context

        # Get conversation memory
        memory_text = ""
        summary_text = ""

        recent_messages = await self.conversation.get_recent_messages(
            username, session_id, limit=self.settings.conversation_window_size
        )
        if recent_messages:
            memory_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_messages[-4:]
            )

        user_summary = await self.summary.get_summary(username)
        if user_summary:
            summary_text = user_summary

        context = assemble_context(
            chunks=results,
            max_chars=self.settings.max_context_chars,
            memory_text=memory_text,
            summary_text=summary_text,
        )

        # --- Step 8: Generate Grounded Answer ---
        from app.core.constants import GROUNDED_ANSWER_PROMPT

        messages = [
            {"role": "system", "content": GROUNDED_ANSWER_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
        ]

        answer, usage = await self.llm_service.generate(
            messages=messages,
            max_tokens=self.settings.max_completion_tokens,
            temperature=0.0,
        )

        # --- Step 9: Verify Answer ---
        from app.guardrails.verification import verify_answer

        verdict_obj = await verify_answer(question, context, answer, self.llm_service)
        verdict = verdict_obj.model_dump() if hasattr(verdict_obj, "model_dump") else verdict_obj.dict()

        # --- Step 10: Apply Refusal Policy ---
        from app.guardrails.policy import apply_refusal_policy

        final_answer, was_refused = apply_refusal_policy(
            verdict=verdict,
            answer=answer,
            confidence_threshold=self.settings.verification_confidence_threshold,
        )

        final_results = results
        if was_refused:
            # Try query rewrite + retry
            final_answer, verdict, retry_results = await self._retry_with_rewrite(
                question, context, results, trace_id
            )
            if retry_results:
                final_results = retry_results

        confidence = float(verdict.get("confidence", 0.0))

        # --- Step 11: Build Source References ---
        sources = []
        for r in final_results[:3]:
            meta = r.get("metadata") or {}
            sources.append({
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", 0),
                "score": r.get("score", 0.0),
                "text_preview": r.get("text", "")[:100],
            })

        # --- Step 12: Store Conversation + Cache ---
        await self.conversation.add_message(username, session_id, "user", question)
        await self.conversation.add_message(username, session_id, "assistant", final_answer)

        # Check if we should summarize
        msg_count = len(
            await self.conversation.get_full_history(username, session_id)
        )
        from app.memory.summary import SummaryManager

        if SummaryManager.should_summarize(msg_count, self.settings.summary_interval):
            try:
                history = await self.conversation.get_full_history(username, session_id)
                history_text = "\n".join(
                    f"{m['role']}: {m['content']}" for m in history
                )
                from app.core.constants import SUMMARY_PROMPT

                summary_messages = [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": history_text},
                ]
                new_summary, _ = await self.llm_service.generate(
                    messages=summary_messages,
                    max_tokens=self.settings.summary_max_tokens,
                    temperature=0.0,
                )
                await self.summary.update_summary(username, new_summary)
            except Exception as e:
                logger.warning("summary_generation_failed", error=str(e))

        # Cache the response
        from app.cache.response_cache import cache_response

        response = self._build_response(
            trace_id=trace_id,
            mode="retrieval",
            answer=final_answer,
            confidence=confidence,
            sources=sources,
            verdict=verdict,
            usage=usage,
            timestamp=timestamp,
            retrieved_contexts=[r.get("text", "") for r in final_results],
        )

        await cache_response(self.cache, cache_key, response, self.settings.cache_ttl_seconds)

        # Record metrics
        latency = time.time() - start_time
        total_tokens = usage.get("total_tokens", 0) if usage else 0
        if self.metrics:
            self.metrics.record_request("retrieval", latency, total_tokens, False)

        # MLflow tracking
        if self.tracker:
            try:
                self.tracker.log_query(
                    question=question,
                    answer=final_answer,
                    confidence=confidence,
                    tokens=total_tokens,
                    latency=latency,
                    cache_hit=False,
                    mode="retrieval",
                    sources=[s["source"] for s in sources],
                )
            except Exception as e:
                logger.warning("mlflow_tracking_failed", error=str(e))

        logger.info(
            "query_processed",
            trace_id=trace_id,
            mode="retrieval",
            confidence=confidence,
            latency_ms=round(latency * 1000),
            tokens=total_tokens,
        )

        return response

    async def _retry_with_rewrite(
        self,
        question: str,
        original_context: str,
        original_results: list,
        trace_id: str,
    ) -> tuple:
        """Retry with query rewrite when first attempt fails verification."""
        from app.core.constants import GROUNDED_ANSWER_PROMPT, QUERY_REWRITE_PROMPT
        from app.guardrails.verification import verify_answer
        from app.utils.json_parser import parse_json_object

        logger.info("retrying_with_rewrite", trace_id=trace_id)

        # Rewrite query
        rewrite_messages = [
            {"role": "system", "content": QUERY_REWRITE_PROMPT},
            {"role": "user", "content": f"Question:\n{question}"},
        ]
        rewrite_text, _ = await self.llm_service.generate(
            messages=rewrite_messages,
            max_tokens=96,
            temperature=0.0,
            response_json=True,
        )
        rewrite_obj = parse_json_object(rewrite_text)
        new_query = str(rewrite_obj.get("query", "")).strip() or question

        # Re-retrieve with a larger candidate pool
        query_vector = self.embedding_service.embed_query(new_query)
        retry_candidate_k = max(12, self.settings.top_k * 2)
        retry_results = self.retriever.search(
            query_vector=query_vector,
            top_k=retry_candidate_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )

        if not retry_results:
            return REFUSAL_RESPONSE, {
                "supported": False,
                "confidence": 0.0,
                "reason": "no_retry_context",
            }, []

        # Rerank & Deduplicate with Cross-Document Correlation
        from app.rag.reranking import rerank_results
        retry_results = rerank_results(retry_results, top_k=self.settings.top_k)

        from app.rag.context import assemble_context

        retry_context = assemble_context(
            chunks=retry_results,
            max_chars=self.settings.max_context_chars,
        )

        # Re-generate
        messages = [
            {"role": "system", "content": GROUNDED_ANSWER_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{retry_context}"},
        ]
        retry_answer, _ = await self.llm_service.generate(
            messages=messages,
            max_tokens=self.settings.max_completion_tokens,
            temperature=0.0,
        )

        # Re-verify
        retry_verdict_obj = await verify_answer(question, retry_context, retry_answer, self.llm_service)
        retry_verdict = retry_verdict_obj.model_dump() if hasattr(retry_verdict_obj, "model_dump") else retry_verdict_obj.dict()

        if (
            retry_verdict.get("supported") is True
            and float(retry_verdict.get("confidence", 0.0))
            >= self.settings.verification_confidence_threshold
        ):
            return retry_answer, retry_verdict, retry_results

        return REFUSAL_RESPONSE, retry_verdict, retry_results

    def _build_response(
        self,
        trace_id: str,
        mode: str,
        answer: str,
        confidence: float,
        timestamp: str,
        sources: list = None,
        verdict: dict = None,
        usage: dict = None,
        cached: bool = False,
        retrieved_contexts: list = None,
    ) -> Dict[str, Any]:
        """Build a standardized response dict."""
        return {
            "trace_id": trace_id,
            "mode": mode,
            "answer": answer,
            "confidence": confidence,
            "sources": sources or [],
            "verdict": verdict
            or {"supported": confidence > 0, "confidence": confidence, "reason": mode},
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cached": cached,
            "timestamp": timestamp,
            "retrieved_contexts": retrieved_contexts or [],
        }
