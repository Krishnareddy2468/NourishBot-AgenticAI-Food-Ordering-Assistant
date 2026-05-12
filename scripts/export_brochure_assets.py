from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML_PATH = ROOT / "docs" / "DZUKKU_PLATFORM_BROCHURE.html"


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def node_text(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            text = clean_text(child.get_text(" ", strip=True))
            if child.name in {"strong", "b"}:
                parts.append(f"<b>{text}</b>")
            elif child.name in {"em", "i"}:
                parts.append(f"<i>{text}</i>")
            else:
                parts.append(text)
    return clean_text(" ".join(parts)).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>").replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")


def image_flowable(img: Tag, html_path: Path):
    src = img.get("src")
    if not src:
        return None
    path = Path(src)
    image_path = path if path.is_absolute() else html_path.parent / path
    if not image_path.exists():
        return None
    max_width = A4[0] - 32 * mm
    try:
        from PIL import Image as PILImage

        with PILImage.open(image_path) as im:
            width, height = im.size
        aspect = height / width if width else 0.55
    except Exception:
        aspect = 0.55
    draw_width = max_width
    draw_height = draw_width * aspect
    if aspect > 1.1:
        draw_width = 72 * mm
        draw_height = draw_width * aspect
    elif draw_height > 95 * mm:
        draw_height = 95 * mm
        draw_width = draw_height / aspect
    return Image(str(image_path), width=draw_width, height=draw_height)


def output_paths(html_path: Path) -> tuple[Path, Path]:
    return html_path.with_suffix(".pdf"), html_path.with_suffix(".doc")


def chrome_path() -> Path | None:
    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_pdf_with_chrome(html_path: Path, pdf_path: Path) -> bool:
    chrome = chrome_path()
    if not chrome:
        return False

    if pdf_path.exists():
        pdf_path.unlink()

    profile = Path(tempfile.mkdtemp(prefix="dzukku-chrome-pdf-"))
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        html_path.resolve().as_uri(),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        process.communicate(timeout=35)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    return pdf_path.exists() and pdf_path.stat().st_size > 1024


def build_pdf(html_path: Path, pdf_path: Path) -> None:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverEyebrow",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#c48a2c"),
            uppercase=True,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=34,
            textColor=colors.HexColor("#7d1821"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#374656"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#7d1821"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CardTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#26313c"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13.5,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10.5,
        )
    )

    story = []
    story.append(Paragraph(clean_text(soup.select_one("h1").get_text(" ", strip=True)), styles["CoverTitle"]))
    story.append(Paragraph(clean_text(soup.select_one(".subtitle").get_text(" ", strip=True)), styles["CoverSubtitle"]))

    stats = []
    for stat in soup.select(".hero-stat"):
        strong = clean_text(stat.find("strong").get_text(" ", strip=True))
        span = clean_text(stat.find("span").get_text(" ", strip=True))
        stats.append(Paragraph(f"<b>{strong}</b><br/>{span}", styles["Small"]))
    story.append(
        Table(
            [stats],
            colWidths=[(A4[0] - 32 * mm) / 4] * 4,
            style=[
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8ea")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e1e8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e1e8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ],
        )
    )
    story.append(Spacer(1, 9 * mm))

    main = soup.find("main")
    for section in main.find_all("section", recursive=False):
        if "page-break" in section.get("class", []):
            story.append(PageBreak())
        title = section.find("h2", recursive=False)
        if title:
            story.append(Paragraph(clean_text(title.get_text(" ", strip=True)), styles["SectionTitle"]))

        for child in section.children:
            if not isinstance(child, Tag) or child.name == "h2":
                continue
            if child.name == "p":
                story.append(Paragraph(node_text(child), styles["Body"]))
            elif child.name == "h3":
                story.append(Paragraph(node_text(child), styles["CardTitle"]))
            elif child.name in {"ul", "ol"}:
                items = [
                    ListItem(Paragraph(node_text(li), styles["Body"]), leftIndent=10)
                    for li in child.find_all("li", recursive=False)
                ]
                story.append(ListFlowable(items, bulletType="1" if child.name == "ol" else "bullet", leftIndent=18))
                story.append(Spacer(1, 3 * mm))
            elif child.name == "table":
                rows = []
                for tr in child.find_all("tr"):
                    row = [Paragraph(node_text(cell), styles["Small"]) for cell in tr.find_all(["th", "td"])]
                    if row:
                        rows.append(row)
                if rows:
                    col_width = (A4[0] - 32 * mm) / max(len(rows[0]), 1)
                    story.append(
                        Table(
                            rows,
                            colWidths=[col_width] * len(rows[0]),
                            repeatRows=1,
                            style=[
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f6")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#22303c")),
                                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9e1e8")),
                                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9e1e8")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ],
                        )
                    )
                    story.append(Spacer(1, 4 * mm))
            elif child.name == "img":
                flowable = image_flowable(child, html_path)
                if flowable:
                    story.append(flowable)
                    story.append(Spacer(1, 3 * mm))
            elif child.name == "div":
                for nested in child.find_all(["img", "h3", "p", "li", "table"], recursive=True):
                    if nested.name == "img":
                        flowable = image_flowable(nested, html_path)
                        if flowable:
                            story.append(flowable)
                            story.append(Spacer(1, 3 * mm))
                        continue
                    if nested.name == "h3":
                        story.append(Paragraph(node_text(nested), styles["CardTitle"]))
                    elif nested.name == "p":
                        story.append(Paragraph(node_text(nested), styles["Body"]))
                    elif nested.name == "li":
                        story.append(Paragraph(f"&bull; {node_text(nested)}", styles["Body"]))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Dzukku Platform Brochure",
        author="Project Dzukku",
    )
    doc.build(story)


def build_word_compatible_doc(html_path: Path, doc_path: Path) -> None:
    shutil.copyfile(html_path, doc_path)


if __name__ == "__main__":
    import sys

    targets = [Path(arg) for arg in sys.argv[1:]] or [DEFAULT_HTML_PATH]
    for target in targets:
        html_path = target if target.is_absolute() else ROOT / target
        pdf_path, doc_path = output_paths(html_path)
        if not build_pdf_with_chrome(html_path, pdf_path):
            build_pdf(html_path, pdf_path)
        build_word_compatible_doc(html_path, doc_path)
        print(f"Generated {pdf_path.relative_to(ROOT)}")
        print(f"Generated {doc_path.relative_to(ROOT)}")
