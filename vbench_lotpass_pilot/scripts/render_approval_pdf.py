#!/usr/bin/env python3
"""Render the Stage A Markdown approval request as a PDF with its contact sheet."""
from __future__ import annotations

import re
import textwrap
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "vbench_lotpass_pilot"
SOURCE = EXP / "selection/approval_request.md"
OUTPUT = EXP / "selection/approval_request.pdf"
CONTACT_SHEET = EXP / "selection/contact_sheet.png"
PAGE_W, PAGE_H = 612, 792
LEFT, RIGHT, TOP, BOTTOM = 45, 45, 750, 42
CONTENT_W = PAGE_W - LEFT - RIGHT
pages: list[bytes] = []
commands: list[str] = []
y = TOP


def new_page() -> None:
    global commands, y
    if commands:
        pages.append("\n".join(commands).encode("ascii"))
    commands = []
    y = TOP


def pdf_hex(text: str) -> str:
    return "<" + text.encode("cp1252", errors="replace").hex().upper() + ">"


def draw_text(text: str, x: float, baseline: float, size: float = 9, bold: bool = False) -> None:
    font = "F2" if bold else "F1"
    commands.append(f"BT /{font} {size} Tf {x:.2f} {baseline:.2f} Td {pdf_hex(text)} Tj ET")


def ensure_room(height: float) -> None:
    if y - height < BOTTOM:
        new_page()


def clean_inline(text: str) -> str:
    text = text.replace("\\|", "|")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip()


def paragraph(text: str, size: float = 9, leading: float = 12, indent: float = 0, bold: bool = False) -> None:
    global y
    text = clean_inline(text)
    avg_char = max(3.5, size * (0.53 if not bold else 0.56))
    width = max(20, int((CONTENT_W - indent) / avg_char))
    lines = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]
    ensure_room(len(lines) * leading + 2)
    for line in lines:
        draw_text(line, LEFT + indent, y, size, bold)
        y -= leading
    y -= 3


def add_contact_sheet() -> None:
    global y
    with Image.open(CONTACT_SHEET) as image:
        width0, height0 = image.size
        scale = min(270 / width0, 450 / height0)
        width, height = width0 * scale, height0 * scale
    if y - height < BOTTOM:
        new_page()
    x = (PAGE_W - width) / 2
    commands.append(f"q {width:.2f} 0 0 {height:.2f} {x:.2f} {y-height:.2f} cm /Im0 Do Q")
    y -= height + 14


def draw_table_row(cell_lines: list[list[str]], height: float, widths: list[float], pad: float,
                   font_size: float, leading: float, is_header: bool = False) -> None:
    global y
    commands.append(f"0.78 G 0.45 w {LEFT} {y-height+pad:.2f} m {LEFT+CONTENT_W} {y-height+pad:.2f} l S 0 G")
    x = LEFT
    baseline = y - pad - font_size
    for col, lines in enumerate(cell_lines):
        for line_no, line in enumerate(lines):
            draw_text(line, x + pad, baseline - line_no * leading, font_size, is_header)
        x += widths[col]
    y -= height


def render_table(rows: list[list[str]]) -> None:
    global y
    if not rows:
        return
    header = rows[0]
    if len(header) == 4:
        widths = [42, 78, 260, CONTENT_W - 380]
    elif len(header) == 3:
        widths = [78, 82, CONTENT_W - 160]
    else:
        widths = [CONTENT_W / len(header)] * len(header)
    pad, font_size, leading = 4, 7.6, 9.3

    def wrap_row(row: list[str]) -> list[list[str]]:
        return [textwrap.wrap(clean_inline(value), width=max(8, int((widths[i]-2*pad)/(font_size*.52))),
                              break_long_words=True, break_on_hyphens=False) or [""]
                for i, value in enumerate(row)]

    def height_for(lines: list[list[str]]) -> float:
        return max(map(len, lines)) * leading + 2 * pad

    for index, row in enumerate(rows):
        lines = wrap_row(row)
        height = height_for(lines)
        if y - height < BOTTOM:
            new_page()
            if index > 0:
                head = wrap_row(header)
                draw_table_row(head, height_for(head), widths, pad, font_size, leading, True)
        draw_table_row(lines, height, widths, pad, font_size, leading, index == 0)


def create_pdf() -> None:
    global y
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            y -= 3
            index += 1
            continue
        if line.startswith("|"):
            table = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell or "-") for cell in cells):
                    table.append(cells)
                index += 1
            render_table(table)
            y -= 5
            continue
        if line.startswith("- Contact sheet:"):
            paragraph("Contact sheet: canonical generator-input previews", size=8.7, leading=11)
            add_contact_sheet()
        elif line.startswith("## "):
            y -= 4
            paragraph(line[3:], size=12, leading=15, bold=True)
        elif line.startswith("# "):
            y -= 4
            paragraph(line[2:], size=16, leading=20, bold=True)
        elif line.startswith("- "):
            paragraph("• " + line[2:], size=8.7, leading=11, indent=8)
        elif line.startswith("**") and line.endswith("**"):
            paragraph(line.strip("*"), size=9, leading=12, bold=True)
        else:
            paragraph(line, size=9, leading=12)
        index += 1
    if commands:
        pages.append("\n".join(commands).encode("ascii"))

    with Image.open(CONTACT_SHEET) as image:
        image = image.convert("RGB")
        image_w, image_h = image.size
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG", quality=88, optimize=True)
        image_bytes = jpeg.getvalue()

    objects: list[bytes | None] = [None]

    def reserve() -> int:
        objects.append(None)
        return len(objects) - 1

    catalog_id, pages_id, regular_id, bold_id, image_id = [reserve() for _ in range(5)]
    page_ids = []
    for content in pages:
        content_id, page_id = reserve(), reserve()
        page_ids.append(page_id)
        objects[content_id] = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        objects[page_id] = (f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                            f"/Resources << /Font << /F1 {regular_id} 0 R /F2 {bold_id} 0 R >> "
                            f"/XObject << /Im0 {image_id} 0 R >> >> /Contents {content_id} 0 R >>").encode()
    objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    objects[pages_id] = f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {len(page_ids)} >>".encode()
    objects[regular_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objects[bold_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    objects[image_id] = (f"<< /Type /XObject /Subtype /Image /Width {image_w} /Height {image_h} /ColorSpace /DeviceRGB "
                         f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(image_bytes)} >>\nstream\n").encode() + image_bytes + b"\nendstream"

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj_id in range(1, len(objects)):
        offsets.append(len(pdf))
        pdf += f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects)}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    OUTPUT.write_bytes(pdf)
    print(f"Rendered {OUTPUT} ({len(pages)} pages; embedded contact sheet {image_w}x{image_h})")


if __name__ == "__main__":
    create_pdf()
