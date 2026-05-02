"""Convert A2A message parts → OpenAI Responses-API content blocks.

Handles text, images (jpg/png), PDFs (text-extracted), plain text files, and
videos (frame-sampled) — same set the green agent advertises in its agent
card. PDF text extraction uses pypdf; video frames use opencv at adaptive
intervals capped at MAX_VIDEO_FRAMES.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass

from a2a.types import FilePart, FileWithBytes, FileWithUri, Part, TextPart
from PIL import Image
from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_IMAGE_EDGE = int(os.environ.get("MAX_IMAGE_EDGE", "1568"))  # downscale very large images
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "24"))


@dataclass
class AgentInput:
    """Aggregated input ready for the OpenAI Responses API."""

    text_blocks: list[str]            # text from TextPart + extracted PDF/text
    images: list[tuple[str, bytes]]   # (display_name, JPEG bytes)
    file_summary: list[str]           # human-readable list of attached files
    original_sizes: dict[str, tuple[int, int]]  # original image size before downscale
    derived_views: list[str]          # machine-generated image notes for the model


def _coerce_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        # strip whitespace base64
        return base64.b64decode(data.replace(" ", "").replace("\n", "").replace("\r", ""))
    raise ValueError(f"Unsupported file data type: {type(data)}")


def _image_to_jpeg(raw: bytes, max_edge: int = MAX_IMAGE_EDGE) -> tuple[bytes, tuple[int, int]]:
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    orig_size = img.size
    w, h = orig_size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), orig_size


def _extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            t = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            t = ""
            logger.warning("PDF page %d extraction failed: %s", i, e)
        if t.strip():
            pages.append(f"--- Page {i} ---\n{t}")
    return "\n\n".join(pages)


def _extract_video_frames(raw: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> list[bytes]:
    """Sample up to max_frames evenly spaced JPEG frames from an MP4 blob."""
    try:
        import cv2  # imported lazily so the agent runs without opencv if no videos
        import numpy as np  # noqa: F401
    except ImportError:
        logger.warning("opencv not installed; skipping video frame extraction")
        return []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        n = min(max_frames, max(1, total))
        idxs = [int(i * total / n) for i in range(n)]
        frames: list[bytes] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            frames.append(buf.getvalue())
        cap.release()
        return frames
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def parts_to_input(parts: list[Part]) -> AgentInput:
    text_blocks: list[str] = []
    images: list[tuple[str, bytes]] = []
    summary: list[str] = []
    original_sizes: dict[str, tuple[int, int]] = {}

    for part in parts:
        root = part.root if hasattr(part, "root") else part

        if isinstance(root, TextPart):
            text_blocks.append(root.text)
            continue

        if isinstance(root, FilePart):
            file_obj = root.file
            if isinstance(file_obj, FileWithUri):
                logger.warning("FileWithUri not supported (no fetcher implemented): %s", file_obj.uri)
                continue
            if not isinstance(file_obj, FileWithBytes):
                logger.warning("Unknown file payload type: %s", type(file_obj))
                continue

            name = file_obj.name or "file"
            mime = (file_obj.mime_type or "").lower()
            try:
                raw = _coerce_bytes(file_obj.bytes)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to decode bytes for %s: %s", name, e)
                continue

            ext = os.path.splitext(name)[1].lower()

            if mime.startswith("image/") or ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
                try:
                    jpeg, orig_size = _image_to_jpeg(raw)
                    images.append((name, jpeg))
                    original_sizes[name] = orig_size
                    summary.append(f"image: {name}")
                except Exception as e:  # noqa: BLE001
                    logger.warning("Image decode failed for %s: %s", name, e)
                continue

            if mime == "application/pdf" or ext == ".pdf":
                try:
                    text = _extract_pdf_text(raw)
                    text_blocks.append(f"=== PDF: {name} ===\n{text}")
                    summary.append(f"pdf: {name} ({len(text)} chars)")
                except Exception as e:  # noqa: BLE001
                    logger.warning("PDF extract failed for %s: %s", name, e)
                continue

            if mime.startswith("text/") or ext in (".txt", ".md", ".csv", ".json"):
                try:
                    text_blocks.append(f"=== TEXT: {name} ===\n{raw.decode('utf-8', errors='replace')}")
                    summary.append(f"text: {name}")
                except Exception as e:  # noqa: BLE001
                    logger.warning("Text decode failed for %s: %s", name, e)
                continue

            if mime.startswith("video/") or ext in (".mp4", ".mov", ".avi"):
                frames = _extract_video_frames(raw)
                for i, fb in enumerate(frames):
                    frame_name = f"{name}#frame{i}"
                    images.append((frame_name, fb))
                    try:
                        with Image.open(io.BytesIO(fb)) as img:
                            original_sizes[frame_name] = img.size
                    except Exception:  # noqa: BLE001
                        pass
                summary.append(f"video: {name} ({len(frames)} frames sampled)")
                continue

            logger.warning("Unhandled file type: name=%s mime=%s", name, mime)

    return AgentInput(
        text_blocks=text_blocks,
        images=images,
        file_summary=summary,
        original_sizes=original_sizes,
        derived_views=[],
    )


def images_to_responses_blocks(images: list[tuple[str, bytes]]) -> list[dict]:
    """Build OpenAI Responses-API input_image content blocks from JPEG images."""
    blocks: list[dict] = []
    for name, jpeg in images:
        b64 = base64.b64encode(jpeg).decode("ascii")
        blocks.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail": "high",
            }
        )
    return blocks
