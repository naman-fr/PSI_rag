"""
Multimodal Vision-RAG Pipeline Orchestrator.

Coordinates the full flow:
1. Input validation & safety guardrails
2. Intent classification (greeting vs visual query vs adversarial)
3. Cache check (keyed by query text + image ID)
4. Image pre-processing, OCR extraction, and visual tagging (if image is present)
5. Dual-index vector retrieval (text SLA/tariffs index + visual chunk index)
6. Combined reranking & deduplication
7. Visual-aware context assembly with user memory & rolling summaries
8. Multimodal answer generation (using Gemini / Groq Vision)
9. Post-generation visual grounding verification (hallucination checker)
10. Refusal, rewrite, and retry policy
11. Session memory updates & response caching
"""

import logging
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from PIL import Image
from langsmith import traceable

from app.core.config import get_settings
from app.core.constants import REFUSAL_RESPONSE
from app.core.logging import get_logger

logger = get_logger("orchestrator")


class RAGOrchestrator:
    """Orchestrates the visual and textual RAG pipeline for user queries."""

    def __init__(
        self,
        embedding_service,  # Standard text embedding service
        retriever,          # Dual FAISS retriever
        llm_service,        # Groq/Gemini text services (fallback)
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

        # Initialize vision specific services
        from app.vision.embeddings.clip_embedder import MultimodalEmbedder
        from app.vision.embeddings.ocr_extractor import OCRExtractor
        from app.vision.answer_generation.answer_generator import MultimodalGenerator
        from app.vision.hallucination_checks.hallucination_checker import VisualHallucinationChecker
        from app.cache.image_feature_cache import ImageFeatureCache

        self.multimodal_embedder = MultimodalEmbedder()
        self.ocr_extractor = OCRExtractor()
        self.multimodal_generator = MultimodalGenerator()
        self.hallucination_checker = VisualHallucinationChecker()
        self.image_feature_cache = ImageFeatureCache(self.cache)

    @traceable(name="multimodal_answer_question")
    async def process_query(
        self,
        question: str,
        username: str,
        session_id: Optional[str] = None,
        image_id: Optional[str] = None,
        image_url: Optional[str] = None,
        raw_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Process a user text and/or image query through the multimodal RAG pipeline.
        """
        start_time = time.time()
        trace_id = str(uuid.uuid4())[:12]
        session_id = session_id or f"session_{username}"
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "processing_multimodal_query",
            trace_id=trace_id,
            username=username,
            image_id=image_id,
            has_image_url=image_url is not None,
            has_raw_image=raw_image is not None,
            question_length=len(question),
        )

        # --- Step 1: Input Guardrails ---
        from app.guardrails.input_guard import detect_injection, sanitize_input

        question = sanitize_input(question)

        if detect_injection(question):
            logger.warning("adversarial_input_detected", trace_id=trace_id)
            return self._build_response(
                trace_id=trace_id,
                mode="guardrail_refusal",
                answer="I can only answer questions about the provided documents and images. Please ask a relevant question.",
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
                answer="I can only answer questions about the provided documents and images.",
                confidence=0.0,
                timestamp=timestamp,
            )

        # Allow greetings to skip RAG if no image is uploaded
        if intent == "greeting" and not (image_id or image_url or raw_image):
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

        effective_image_id = image_id or (hash(image_url) if image_url else "") or ""
        if isinstance(effective_image_id, int):
            effective_image_id = str(effective_image_id)
            
        cache_key = make_cache_key(question, image_id=effective_image_id, prompt_version=self.settings.vision_model)
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

        # --- Step 4: Process Image Input ---
        query_image = raw_image
        ocr_text = ""
        image_tags = []
        image_vector = None

        # Resolve image from URL or pre-existing ID if needed
        if not query_image:
            if image_url:
                try:
                    from app.vision.ingestion.image_loader import load_image_from_url, preprocess_image
                    query_image = preprocess_image(load_image_from_url(image_url))
                except Exception as img_err:
                    logger.warning("Failed to fetch image from URL", url=image_url, error=str(img_err))
            elif image_id:
                # Check if we have cached features
                cached_features = await self.image_feature_cache.get_features(image_id)
                if cached_features:
                    image_vector, ocr_text, image_tags = cached_features
                    logger.info("Retrieved image features from cache", image_id=image_id, tags=image_tags)
                else:
                    logger.warning("image_id provided but features not cached, trying local resolve", image_id=image_id)

        # Process the image to extract features if we have one and features aren't cached yet
        if query_image and image_vector is None:
            # Generate ID if missing
            if not effective_image_id:
                effective_image_id = str(uuid.uuid4())[:12]
            
            # Embed image
            image_vector = self.multimodal_embedder.embed_image(query_image)
            # Run OCR
            ocr_text = await self.ocr_extractor.extract_text(query_image)
            # Run Object tagging
            image_tags = await self.ocr_extractor.detect_objects_and_tags(query_image)

            # Store in feature cache
            await self.image_feature_cache.set_features(
                image_id=effective_image_id,
                embedding=image_vector,
                ocr_text=ocr_text,
                tags=image_tags,
                ttl=self.settings.cache_ttl_seconds * 2
            )
            logger.info("Precomputed and cached image features", image_id=effective_image_id)

        # --- Step 5: Multi-modal Vector Search ---
        # 1. Search text documents using standard text embeddings of question
        text_query_vector = self.embedding_service.embed_query(question)
        text_results = self.retriever.search_text(
            query_vector=text_query_vector,
            top_k=self.settings.top_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )

        # 2. Search visual database
        visual_results = []
        if image_vector is not None:
            # Query-by-Image: search visual assets similar to the uploaded image
            visual_results = self.retriever.search_visual(
                query_vector=image_vector,
                top_k=self.settings.top_k,
                score_threshold=self.settings.retrieval_score_threshold,
            )
        else:
            # Query-by-Text: search visual assets using the text query embedded in multimodal space
            mm_text_vector = self.multimodal_embedder.embed_text(question)
            visual_results = self.retriever.search_visual(
                query_vector=mm_text_vector,
                top_k=self.settings.top_k,
                score_threshold=self.settings.retrieval_score_threshold,
            )

        # Combine results
        combined_results = text_results + visual_results

        # --- Step 6: Retrieval Quality Gate ---
        from app.guardrails.retrieval_gate import check_retrieval_quality

        gate_passed, gate_reason = check_retrieval_quality(
            combined_results, min_score=self.settings.retrieval_score_threshold, min_chunks=1
        )

        # Strict gate for text reference, but if we have an image, visual context takes precedence
        if not gate_passed and query_image is None:
            logger.info("retrieval_gate_failed", trace_id=trace_id, reason=gate_reason)
            return self._build_response(
                trace_id=trace_id,
                mode="guardrail_refusal",
                answer=REFUSAL_RESPONSE,
                confidence=0.0,
                verdict={"supported": False, "confidence": 0.0, "reason": gate_reason},
                timestamp=timestamp,
            )

        # --- Step 7: Reranking & Deduplication ---
        from app.vision.reranking.score_refinement import rerank_results

        refined_results = rerank_results(combined_results, top_k=self.settings.top_k)

        # --- Step 8: Assemble Context with Memory ---
        from app.rag.context import assemble_context

        # Get conversation memory
        memory_text = ""
        summary_text = ""

        recent_messages = await self.conversation.get_recent_messages(
            username, session_id, limit=self.settings.conversation_window_size
        )
        if recent_messages:
            # Include image_id context in conversational history if present
            memory_text = "\n".join(
                f"{m['role']}: {m['content']} [image_id: {m.get('image_id', 'none')}]"
                for m in recent_messages[-4:]
            )

        user_summary = await self.summary.get_summary(username)
        if user_summary:
            summary_text = user_summary

        context = assemble_context(
            chunks=refined_results,
            max_chars=self.settings.max_context_chars,
            memory_text=memory_text,
            summary_text=summary_text,
        )

        # --- Step 9: Generate Answer ---
        answer, usage = await self.multimodal_generator.generate(
            question=question,
            context=context,
            image=query_image,
            ocr_text=ocr_text,
            detected_objects=image_tags,
            max_tokens=self.settings.max_completion_tokens,
            temperature=0.0,
        )

        # --- Step 10: Grounding Verification ---
        verdict_obj = await self.hallucination_checker.verify(
            question=question,
            context=context,
            answer=answer,
            ocr_text=ocr_text,
            detected_objects=image_tags,
        )
        verdict = verdict_obj.model_dump() if hasattr(verdict_obj, "model_dump") else verdict_obj.dict()

        # --- Step 11: Apply Refusal Policy & Retry ---
        from app.guardrails.policy import apply_refusal_policy

        final_answer, was_refused = apply_refusal_policy(
            verdict=verdict,
            answer=answer,
            confidence_threshold=self.settings.verification_confidence_threshold,
        )

        if was_refused:
            logger.info("verification_failed_trigger_rewrite", trace_id=trace_id, verdict=verdict)
            # Attempt a visual-specific query rewrite & retry
            final_answer, verdict = await self._retry_with_rewrite(
                question, context, query_image, ocr_text, image_tags, trace_id
            )

        confidence = float(verdict.get("confidence", 0.0))

        # --- Step 12: Build Source References ---
        sources = []
        for r in refined_results[:3]:
            meta = r.get("metadata") or {}
            sources.append({
                "source": meta.get("source") or meta.get("image_id") or "unknown",
                "chunk_id": meta.get("chunk_id") or meta.get("tile_index") or 0,
                "score": r.get("score", 0.0),
                "text_preview": r.get("text", "")[:100],
                "image_id": meta.get("image_id"),
                "tile_index": meta.get("tile_index"),
            })

        # --- Step 13: Store Conversation + Cache ---
        await self.conversation.add_message(
            username=username,
            session_id=session_id,
            role="user",
            content=question,
            image_id=effective_image_id or None
        )
        await self.conversation.add_message(
            username=username,
            session_id=session_id,
            role="assistant",
            content=final_answer,
            image_id=None
        )

        # Summarization interval check
        history = await self.conversation.get_full_history(username, session_id)
        msg_count = len(history)
        from app.memory.summary import SummaryManager

        if SummaryManager.should_summarize(msg_count, self.settings.summary_interval):
            try:
                history_text = "\n".join(
                    f"{m['role']}: {m['content']} [image_id: {m.get('image_id', 'none')}]"
                    for m in history
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
            except Exception as summary_err:
                logger.warning("summary_generation_failed", error=str(summary_err))

        # Build response payload
        response = self._build_response(
            trace_id=trace_id,
            mode="multimodal_retrieval" if query_image is not None else "retrieval",
            answer=final_answer,
            confidence=confidence,
            sources=sources,
            verdict=verdict,
            usage=usage,
            timestamp=timestamp,
        )

        # Cache the response
        from app.cache.response_cache import cache_response
        await cache_response(self.cache, cache_key, response, self.settings.cache_ttl_seconds)

        # Record metrics
        latency = time.time() - start_time
        total_tokens = usage.get("total_tokens", 0) if usage else 0
        if self.metrics:
            self.metrics.record_request("multimodal_retrieval" if query_image else "retrieval", latency, total_tokens, False)

        # MLflow experiment log
        if self.tracker:
            try:
                self.tracker.log_query(
                    question=question,
                    answer=final_answer,
                    confidence=confidence,
                    tokens=total_tokens,
                    latency=latency,
                    cache_hit=False,
                    mode="multimodal_retrieval" if query_image else "retrieval",
                    sources=[s["source"] for s in sources],
                )
            except Exception as mlflow_err:
                logger.warning("mlflow_tracking_failed", error=str(mlflow_err))

        logger.info(
            "query_processed",
            trace_id=trace_id,
            confidence=confidence,
            latency_ms=round(latency * 1000),
            tokens=total_tokens,
        )

        return response

    async def _retry_with_rewrite(
        self,
        question: str,
        original_context: str,
        query_image: Optional[Image.Image],
        ocr_text: Optional[str],
        image_tags: Optional[list[str]],
        trace_id: str,
    ) -> tuple:
        """Visual-aware rewrite and retry policy."""
        from app.core.constants import QUERY_REWRITE_PROMPT
        from app.utils.json_parser import parse_json_object

        logger.info("retry_with_rewrite_visual", trace_id=trace_id)

        # Rewrite query to focus on visual keywords
        rewrite_payload = (
            f"Question:\n{question}\n\n"
            f"Image Tags:\n{', '.join(image_tags or [])}\n"
            f"OCR Layout Preview:\n{ocr_text[:300] if ocr_text else 'none'}\n"
        )
        rewrite_messages = [
            {"role": "system", "content": QUERY_REWRITE_PROMPT},
            {"role": "user", "content": rewrite_payload},
        ]
        
        try:
            rewrite_text, _ = await self.llm_service.generate(
                messages=rewrite_messages,
                max_tokens=96,
                temperature=0.0,
                response_json=True,
            )
            rewrite_obj = parse_json_object(rewrite_text)
            new_query = str(rewrite_obj.get("query", "")).strip() or question
        except Exception:
            new_query = question

        # Re-retrieve with new query
        text_query_vector = self.embedding_service.embed_query(new_query)
        retry_text_results = self.retriever.search_text(
            query_vector=text_query_vector,
            top_k=self.settings.top_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )

        # Visual search with rewritten query embedded in multimodal space
        mm_text_vector = self.multimodal_embedder.embed_text(new_query)
        retry_visual_results = self.retriever.search_visual(
            query_vector=mm_text_vector,
            top_k=self.settings.top_k,
            score_threshold=self.settings.retrieval_score_threshold,
        )

        combined_results = retry_text_results + retry_visual_results
        if not combined_results and query_image is None:
            return REFUSAL_RESPONSE, {
                "supported": False,
                "confidence": 0.0,
                "reason": "no_context_on_retry",
            }

        from app.rag.context import assemble_context
        from app.vision.reranking.score_refinement import rerank_results

        refined_retry_results = rerank_results(combined_results, top_k=self.settings.top_k)
        retry_context = assemble_context(
            chunks=refined_retry_results,
            max_chars=self.settings.max_context_chars,
        )

        # Re-generate
        retry_answer, _ = await self.multimodal_generator.generate(
            question=question,
            context=retry_context,
            image=query_image,
            ocr_text=ocr_text,
            detected_objects=image_tags,
            max_tokens=self.settings.max_completion_tokens,
            temperature=0.0,
        )

        # Re-verify
        retry_verdict_obj = await self.hallucination_checker.verify(
            question=question,
            context=retry_context,
            answer=retry_answer,
            ocr_text=ocr_text,
            detected_objects=image_tags,
        )
        retry_verdict = retry_verdict_obj.model_dump() if hasattr(retry_verdict_obj, "model_dump") else retry_verdict_obj.dict()

        if (
            retry_verdict.get("supported") is True
            and float(retry_verdict.get("confidence", 0.0))
            >= self.settings.verification_confidence_threshold
        ):
            logger.info("retry_successful", trace_id=trace_id)
            return retry_answer, retry_verdict

        logger.warning("retry_failed_verification", trace_id=trace_id, verdict=retry_verdict)
        return REFUSAL_RESPONSE, retry_verdict

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
        }
