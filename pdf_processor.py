#!/usr/bin/env python3
"""
ColPali Long-Running HTTP Server
Supports Apple Silicon M4 with MPS acceleration
"""

import asyncio
import uuid
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator

from PIL import Image
import fitz  # PyMuPDF


@dataclass
class StreamingProgress:
    task_id: str
    progress: float  # 0.0 to 100.0
    current_step: str
    step_num: int
    total_steps: int
    details: str = ""
    eta_seconds: Optional[int] = None
    throughput: Optional[str] = None
    error: Optional[str] = None
    results: Optional[List[Dict]] = None  # For storing search results

    def to_dict(self):
        return asdict(self)


class PDFProcessor:
    """Handles PDF processing and page extraction"""

    @staticmethod
    async def extract_pages(file_path: str) -> AsyncGenerator[StreamingProgress, None]:
        """Extract pages from PDF with streaming progress"""
        task_id = f"extract_{uuid.uuid4().hex[:8]}"

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Opening PDF file",
            step_num=1,
            total_steps=3,
            details=f"Loading {file_path}",
        )

        await asyncio.sleep(0.5)

        # Open PDF
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
        except Exception as e:
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Error opening PDF",
                step_num=1,
                total_steps=3,
                error=str(e),
            )
            return

        yield StreamingProgress(
            task_id=task_id,
            progress=20.0,
            current_step=f"Processing {total_pages} pages",
            step_num=2,
            total_steps=3,
            details="Extracting images and text",
        )

        images = []
        metadata = []

        for page_num in range(total_pages):
            page = doc[page_num]

            # Extract image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)

            # Extract text
            text_content = page.get_text()

            metadata.append(
                {
                    "page_num": page_num + 1,
                    "doc_name": Path(file_path).stem,
                    "text_content": text_content,
                }
            )

            # Update progress
            page_progress = 20.0 + (page_num + 1) / total_pages * 70.0
            yield StreamingProgress(
                task_id=task_id,
                progress=page_progress,
                current_step=f"Extracted page {page_num + 1}/{total_pages}",
                step_num=2,
                total_steps=3,
                details=f"Image: {image.size}, Text: {len(text_content)} chars",
            )

            await asyncio.sleep(0.1)  # Yield control

        doc.close()

        yield StreamingProgress(
            task_id=task_id,
            progress=100.0,
            current_step="PDF extraction complete",
            step_num=3,
            total_steps=3,
            details=f"Extracted {len(images)} pages successfully",
        )
