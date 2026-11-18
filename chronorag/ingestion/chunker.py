import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from chronorag.ingestion.reader import ParsedBlock


@dataclass
class DocumentChunk:
    text: str
    page_number: Optional[int]
    section_heading: Optional[str]
    chunk_index: int
    metadata: Dict[str, Any]


class StructureAwareChunker:
    """
    Chunker that preserves section headings, paragraph boundaries, and page numbers
    while applying sliding-window token/character overlap.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_blocks(self, blocks: List[ParsedBlock]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        current_text_parts: List[str] = []
        current_char_len = 0
        current_page = blocks[0].page_number if blocks else 1
        current_heading = blocks[0].section_heading if blocks else None

        for block in blocks:
            # If section heading changed or page changed significantly, consider splitting
            heading = block.section_heading or current_heading
            page = block.page_number or current_page
            block_text = block.text.strip()

            if not block_text:
                continue

            # If block itself exceeds chunk_size, split by sentences
            if len(block_text) > self.chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", block_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if current_char_len + len(sentence) > self.chunk_size and current_text_parts:
                        # Flush current chunk
                        full_chunk_text = " ".join(current_text_parts)
                        chunks.append(
                            DocumentChunk(
                                text=full_chunk_text,
                                page_number=current_page,
                                section_heading=current_heading,
                                chunk_index=chunk_idx,
                                metadata={"type": "structured_paragraph"},
                            )
                        )
                        chunk_idx += 1

                        # Compute overlap parts
                        overlap_target = self.chunk_overlap
                        overlap_parts = []
                        acc_len = 0
                        for part in reversed(current_text_parts):
                            overlap_parts.insert(0, part)
                            acc_len += len(part)
                            if acc_len >= overlap_target:
                                break
                        current_text_parts = overlap_parts
                        current_char_len = sum(len(p) for p in current_text_parts)

                    current_text_parts.append(sentence)
                    current_char_len += len(sentence)
                    current_page = page
                    current_heading = heading
            else:
                if current_char_len + len(block_text) > self.chunk_size and current_text_parts:
                    full_chunk_text = " ".join(current_text_parts)
                    chunks.append(
                        DocumentChunk(
                            text=full_chunk_text,
                            page_number=current_page,
                            section_heading=current_heading,
                            chunk_index=chunk_idx,
                            metadata={"type": "structured_block"},
                        )
                    )
                    chunk_idx += 1

                    # Keep overlap
                    overlap_target = self.chunk_overlap
                    overlap_parts = []
                    acc_len = 0
                    for part in reversed(current_text_parts):
                        overlap_parts.insert(0, part)
                        acc_len += len(part)
                        if acc_len >= overlap_target:
                            break
                    current_text_parts = overlap_parts
                    current_char_len = sum(len(p) for p in current_text_parts)

                current_text_parts.append(block_text)
                current_char_len += len(block_text)
                current_page = page
                current_heading = heading

        # Flush trailing chunk
        if current_text_parts:
            full_chunk_text = " ".join(current_text_parts)
            chunks.append(
                DocumentChunk(
                    text=full_chunk_text,
                    page_number=current_page,
                    section_heading=current_heading,
                    chunk_index=chunk_idx,
                    metadata={"type": "structured_block"},
                )
            )

        return chunks
