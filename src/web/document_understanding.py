import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

try:
    from mistralai import Mistral, DocumentURLChunk
    from mistralai.extra import response_format_from_pydantic_model
    MISTRALAI_AVAILABLE = True
except ImportError:
    MISTRALAI_AVAILABLE = False

class BBoxImageAnnotation(BaseModel):
    image_type: str = Field(..., description="The type of the image.")
    short_description: str = Field(..., description="A description in english describing the image.")
    summary: str = Field(..., description="Summarize the image.")

class DocumentAnnotation(BaseModel):
    language: str
    chapter_titles: list[str]
    urls: list[str]

    # You can extend this model as needed

class DocumentUnderstanding:
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-small-latest"):
        if not MISTRALAI_AVAILABLE:
            raise ImportError("mistralai package is not installed. Please install with: pip install mistralai")
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set.")
        self.model = model
        self.client = Mistral(api_key=self.api_key)

    def ask_document(self, question: str, document_url: str) -> str:
        """
        Ask a question about a document (PDF or other supported type) using Mistral LLM.
        :param question: The question to ask about the document
        :param document_url: The URL (or signed URL) of the document
        :return: The answer from the LLM
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "document_url", "document_url": document_url}
                ]
            }
        ]
        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )
        return chat_response.choices[0].message.content

    def upload_document(self, file_path: str) -> str:
        """
        Upload a local document and get a signed URL for use in ask_document.
        :param file_path: Path to the local file
        :return: Signed URL for the uploaded document
        """
        uploaded_pdf = self.client.files.upload(
            file={
                "file_name": os.path.basename(file_path),
                "content": open(file_path, "rb"),
            },
            purpose="ocr"
        )
        signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
        return signed_url.url

    def process_with_annotations(
        self,
        document_url: str,
        pages: list[int] = None,
        bbox_annotation_model: BaseModel = None,
        document_annotation_model: BaseModel = None,
        include_image_base64: bool = True
    ) -> dict:
        """
        Process a document with optional bbox and document annotation formats.
        :param document_url: URL to the document
        :param pages: List of page numbers (optional)
        :param bbox_annotation_model: Pydantic model for bbox annotation (optional)
        :param document_annotation_model: Pydantic model for document annotation (optional)
        :param include_image_base64: Whether to include image base64 in response
        :return: API response as dict
        """
        kwargs = {
            "model": "mistral-ocr-latest",
            "document": DocumentURLChunk(document_url=document_url),
            "include_image_base64": include_image_base64
        }
        if pages is not None:
            kwargs["pages"] = pages
        if bbox_annotation_model is not None:
            kwargs["bbox_annotation_format"] = response_format_from_pydantic_model(bbox_annotation_model)
        if document_annotation_model is not None:
            kwargs["document_annotation_format"] = response_format_from_pydantic_model(document_annotation_model)
        response = self.client.ocr.process(**kwargs)
        return response
