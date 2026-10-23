import pytest
from pydantic import ValidationError
from backend.schemas import ChatRequest, ImageAnalyzeRequest, ImageGenerateRequest
def test_message_is_trimmed(): assert ChatRequest(message=" hello ").message=="hello"
def test_blank_message_rejected():
 with pytest.raises(ValidationError): ChatRequest(message="  ")
def test_unknown_skill_rejected():
 with pytest.raises(ValidationError): ChatRequest(message="hello",skill="wizard")


def test_image_analysis_defaults_and_mode_validation():
    request = ImageAnalyzeRequest(image_data_url="data:image/png;base64,AAAA")
    assert request.mode == "describe"
    assert request.prompt == "Describe this image in detail."
    with pytest.raises(ValidationError):
        ImageAnalyzeRequest(
            image_data_url="data:image/png;base64,AAAA",
            mode="generate",
        )


def test_image_generation_requires_safe_dimensions_and_prompt():
    request = ImageGenerateRequest(prompt=" A lake ", width=512, height=768)
    assert request.prompt == "A lake"
    assert request.seed is None
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="A lake", width=513)
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="   ")
