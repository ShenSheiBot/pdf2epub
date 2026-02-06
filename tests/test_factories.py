"""
Smoke tests for factory functions.

These catch wiring errors early (e.g., mismatched constructor signatures)
without requiring real LLM providers.
"""


def test_create_processing_pipeline_v2_smoke(tmp_path):
    """Factory should build a pipeline with a tracker wired in."""
    from pdf2epub.core.factory_v2 import create_processing_pipeline_v2

    class DummyProcessor:
        @property
        def name(self) -> str:
            return "dummy"

        def build_prompt(self, content, context):
            return f"Process: {content}"

        def clean_response(self, response):
            return response

        def post_process(self, result, context):
            return result

        def get_model_configs(self):
            return [{"provider": "test", "model": "test"}]

    class DummyLLMClient:
        pass

    pipeline = create_processing_pipeline_v2(
        processor=DummyProcessor(),
        output_dir=tmp_path,
        llm_client=DummyLLMClient(),
        config=None,
        task_type="translate",
        use_batch_validation=False,
    )

    assert pipeline is not None

