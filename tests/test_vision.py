import io
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from PIL import Image

from app.vision.ingestion.image_loader import load_image_from_bytes, preprocess_image, to_opencv_format, to_pil_format
from app.vision.chunking.image_tiler import tile_image
from app.vision.embeddings.clip_embedder import MultimodalEmbedder
from app.vision.embeddings.ocr_extractor import OCRExtractor
from app.vision.retrieval.vector_search import DualRetriever
from app.cache.image_feature_cache import ImageFeatureCache
from app.cache.redis_client import InMemoryCache


class TestVisionComponents:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create a simple red test image
        img_bytes = io.BytesIO()
        Image.new("RGB", (200, 200), color="red").save(img_bytes, format="PNG")
        self.sample_bytes = img_bytes.getvalue()
        self.sample_pil = Image.open(io.BytesIO(self.sample_bytes))

    def test_image_loading_and_conversion(self):
        img = load_image_from_bytes(self.sample_bytes)
        assert img.size == (200, 200)
        assert img.mode == "RGB"

        # Convert to OpenCV format
        cv_img = to_opencv_format(img)
        assert cv_img.shape == (200, 200, 3)

        # Convert back to PIL
        pil_img = to_pil_format(cv_img)
        assert pil_img.size == (200, 200)

    def test_preprocess_image_resize(self):
        large_img = Image.new("RGB", (2000, 1000), color="blue")
        processed = preprocess_image(large_img, max_dim=1024)
        # Ratio 2:1, so width should scale to 1024 and height to 512
        assert processed.size == (1024, 512)

    def test_image_tiling(self):
        large_img = Image.new("RGB", (1024, 1024), color="green")
        tiles = tile_image(large_img, tile_size=512, overlap=64)
        # Should generate multiple overlapping tiles
        assert len(tiles) > 1
        for tile, box in tiles:
            assert tile.size == (512, 512)
            assert len(box) == 4

    @patch("app.vision.embeddings.clip_embedder.genai.Client")
    def test_multimodal_embedder(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock embed response
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 1408
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_response

        embedder = MultimodalEmbedder()
        vec = embedder.embed_text("test question")
        assert vec.shape == (1408,)
        # Check normalized vector
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

        img_vec = embedder.embed_image(self.sample_pil)
        assert img_vec.shape == (1408,)
        assert abs(np.linalg.norm(img_vec) - 1.0) < 1e-4

    @patch("app.vision.embeddings.ocr_extractor.genai.Client")
    @pytest.mark.asyncio
    async def test_ocr_and_tagger(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock generate responses
        mock_ocr_response = MagicMock()
        mock_ocr_response.text = "INVOICE DETAILS:\nTOTAL: $500"
        
        mock_tag_response = MagicMock()
        mock_tag_response.text = "invoice, label, document"

        mock_client.models.generate_content.side_effect = [
            mock_ocr_response,
            mock_tag_response
        ]

        extractor = OCRExtractor()
        ocr_result = await extractor.extract_text(self.sample_pil)
        assert "TOTAL" in ocr_result

        tags = await extractor.detect_objects_and_tags(self.sample_pil)
        assert tags == ["invoice", "label", "document"]

    @pytest.mark.asyncio
    async def test_dual_retriever(self):
        retriever = DualRetriever(text_dimension=3, visual_dimension=3)
        
        # Build text index
        text_vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        text_meta = [{"text": "SLA Doc 1", "source": "sla"}, {"text": "Tariff Doc 2", "source": "tariff"}]
        await retriever.build_text_index(text_vecs, text_meta)
        
        # Search text
        results = retriever.search_text(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1, score_threshold=0.5)
        assert len(results) == 1
        assert results[0]["text"] == "SLA Doc 1"

        # Build visual index
        visual_vecs = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        visual_meta = [{"text": "Cargo Box", "image_id": "img1"}]
        await retriever.build_visual_index(visual_vecs, visual_meta)

        # Search visual
        v_results = retriever.search_visual(np.array([0.0, 0.0, 1.0], dtype=np.float32), top_k=1, score_threshold=0.5)
        assert len(v_results) == 1
        assert v_results[0]["text"] == "Cargo Box"

        # Dynamic add visual vector
        new_v_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        await retriever.add_visual_vector(new_v_vec, {"text": "Container Door", "image_id": "img2"})
        assert retriever.visual_count == 2

    @pytest.mark.asyncio
    async def test_image_feature_cache(self):
        in_memory_backend = InMemoryCache()
        feature_cache = ImageFeatureCache(in_memory_backend)

        mock_embedding = np.array([0.5, 0.5, 0.707], dtype=np.float32)
        ocr = "SLA agreement text"
        tags = ["label", "stamp"]

        await feature_cache.set_features("img123", mock_embedding, ocr, tags)
        
        cached = await feature_cache.get_features("img123")
        assert cached is not None
        res_emb, res_ocr, res_tags = cached
        
        np.testing.assert_array_almost_equal(res_emb, mock_embedding)
        assert res_ocr == ocr
        assert res_tags == tags
