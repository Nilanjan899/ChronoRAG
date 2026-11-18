import csv
import io
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ParsedBlock:
    text: str
    page_number: Optional[int] = 1
    section_heading: Optional[str] = None
    block_type: str = "paragraph"  # heading, paragraph, table
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentReader:
    """Multi-format document reader preserving page numbers and section hierarchy."""

    @classmethod
    def read_file(cls, file_path: Path) -> List[ParsedBlock]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return cls.read_pdf(file_path)
        elif ext == ".docx":
            return cls.read_docx(file_path)
        elif ext in [".md", ".markdown"]:
            return cls.read_markdown(file_path)
        elif ext in [".csv"]:
            return cls.read_csv(file_path)
        elif ext in [".txt", ""]:
            return cls.read_txt(file_path)
        else:
            # Fallback to plain text
            return cls.read_txt(file_path)

    @classmethod
    def read_pdf(cls, file_path: Path) -> List[ParsedBlock]:
        import fitz  # PyMuPDF

        blocks: List[ParsedBlock] = []
        doc = fitz.open(file_path)
        current_heading: Optional[str] = None

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_text_lines = []
                    is_heading = False
                    for line in block.get("lines", []):
                        line_text = "".join(span.get("text", "") for span in line.get("spans", "")).strip()
                        if line_text:
                            # Simple heuristic: bold or larger font implies heading
                            span = line["spans"][0] if line.get("spans") else {}
                            font_size = span.get("size", 10)
                            font_flags = span.get("flags", 0)
                            if font_size >= 13 or (font_flags & 2):  # bold or big
                                if len(line_text) < 100:
                                    is_heading = True
                            block_text_lines.append(line_text)

                    full_block_text = " ".join(block_text_lines).strip()
                    if full_block_text:
                        if is_heading and len(full_block_text) < 120:
                            current_heading = full_block_text
                            blocks.append(
                                ParsedBlock(
                                    text=full_block_text,
                                    page_number=page_num,
                                    section_heading=current_heading,
                                    block_type="heading",
                                )
                            )
                        else:
                            blocks.append(
                                ParsedBlock(
                                    text=full_block_text,
                                    page_number=page_num,
                                    section_heading=current_heading,
                                    block_type="paragraph",
                                )
                            )
        doc.close()
        return blocks

    @classmethod
    def read_docx(cls, file_path: Path) -> List[ParsedBlock]:
        import docx

        doc = docx.Document(file_path)
        blocks: List[ParsedBlock] = []
        current_heading: Optional[str] = None

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            style_name = p.style.name.lower() if p.style else ""
            if "heading" in style_name:
                current_heading = text
                blocks.append(
                    ParsedBlock(
                        text=text,
                        page_number=1,  # DOCX doesn't have native pages without layout engine
                        section_heading=current_heading,
                        block_type="heading",
                    )
                )
            else:
                blocks.append(
                    ParsedBlock(
                        text=text,
                        page_number=1,
                        section_heading=current_heading,
                        block_type="paragraph",
                    )
                )

        # Handle tables
        for t_idx, table in enumerate(doc.tables):
            table_rows = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_rows.append(row_text)
            if table_rows:
                blocks.append(
                    ParsedBlock(
                        text="\n".join(table_rows),
                        page_number=1,
                        section_heading=current_heading or f"Table {t_idx + 1}",
                        block_type="table",
                    )
                )
        return blocks

    @classmethod
    def read_markdown(cls, file_path: Path) -> List[ParsedBlock]:
        content = file_path.read_text(encoding="utf-8")
        blocks: List[ParsedBlock] = []
        current_heading: Optional[str] = None
        current_para: List[str] = []

        lines = content.splitlines()
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                if current_para:
                    blocks.append(
                        ParsedBlock(
                            text=" ".join(current_para),
                            page_number=1,
                            section_heading=current_heading,
                            block_type="paragraph",
                        )
                    )
                    current_para = []
                continue

            header_match = re.match(r"^(#{1,6})\s+(.*)$", trimmed)
            if header_match:
                if current_para:
                    blocks.append(
                        ParsedBlock(
                            text=" ".join(current_para),
                            page_number=1,
                            section_heading=current_heading,
                            block_type="paragraph",
                        )
                    )
                    current_para = []
                current_heading = header_match.group(2).strip()
                blocks.append(
                    ParsedBlock(
                        text=current_heading,
                        page_number=1,
                        section_heading=current_heading,
                        block_type="heading",
                    )
                )
            else:
                current_para.append(trimmed)

        if current_para:
            blocks.append(
                ParsedBlock(
                    text=" ".join(current_para),
                    page_number=1,
                    section_heading=current_heading,
                    block_type="paragraph",
                )
            )
        return blocks

    @classmethod
    def read_csv(cls, file_path: Path) -> List[ParsedBlock]:
        blocks: List[ParsedBlock] = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            header_str = ", ".join(header)
            blocks.append(
                ParsedBlock(
                    text=f"CSV Columns: {header_str}",
                    page_number=1,
                    section_heading="Header",
                    block_type="heading",
                )
            )
            for row_idx, row in enumerate(reader, start=1):
                row_str = " | ".join(f"{col}: {val}" for col, val in zip(header, row) if val.strip())
                if row_str:
                    blocks.append(
                        ParsedBlock(
                            text=row_str,
                            page_number=1,
                            section_heading="Data",
                            block_type="table",
                            metadata={"row_index": row_idx},
                        )
                    )
        return blocks

    @classmethod
    def read_txt(cls, file_path: Path) -> List[ParsedBlock]:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        paragraphs = re.split(r"\n\s*\n", content)
        blocks: List[ParsedBlock] = []
        for p in paragraphs:
            trimmed = p.strip()
            if trimmed:
                blocks.append(
                    ParsedBlock(
                        text=trimmed,
                        page_number=1,
                        section_heading=None,
                        block_type="paragraph",
                    )
                )
        return blocks
