"""Render the bundled report using reportlab; no model fitting or source edits."""
from datetime import date
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

BASE = Path(__file__).resolve().parents[1]
_date = date.fromisoformat(json.loads((BASE / "reproducibility/report_content.json").read_text(encoding="utf-8"))["date"])
REPORT_DATE = f"{_date.day} {_date:%B %Y}".upper()
NAVY = colors.HexColor("#19374b")
TEAL = colors.HexColor("#087e8b")
INK = colors.HexColor("#263746")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=NAVY, spaceAfter=17))
styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle(name="Subsection", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=TEAL, spaceBefore=11, spaceAfter=6))
styles.add(ParagraphStyle(name="Copy", fontName="Helvetica", fontSize=10, leading=14.2, textColor=INK, spaceAfter=9))
styles.add(ParagraphStyle(name="SmallCopy", fontName="Helvetica", fontSize=8.1, leading=11, textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8, leading=10.6, textColor=INK))
styles.add(ParagraphStyle(name="HeadCell", fontName="Helvetica-Bold", fontSize=8, leading=10.6, textColor=colors.white))


def p(text, style="Copy"):
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])


def footer(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d9e2e8"))
    canvas.line(42, 40, w - 42, 40)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#657787"))
    canvas.drawString(42, 27, f"EMM PROJECT  |  COMPLETED EXPERIMENTS  |  {REPORT_DATE}")
    canvas.drawRightString(w - 42, 27, str(doc.page))
    canvas.restoreState()


def main():
    document = json.loads((BASE / "reproducibility/report_content.json").read_text(encoding="utf-8"))
    story = []
    for i, section in enumerate(document["sections"]):
        if i:
            story.append(PageBreak())
        story.append(p(section["title"], "ReportTitle" if i == 0 else "Section"))
        for block in section["blocks"]:
            kind = block["type"]
            if kind in ("paragraph", "small", "heading"):
                story.append(p(block["text"], {"paragraph":"Copy", "small":"SmallCopy", "heading":"Subsection"}[kind]))
            elif kind == "table":
                rows = [[p(c, "HeadCell" if j == 0 else "Cell") for c in row] for j, row in enumerate(block["rows"])]
                table = Table(rows, colWidths=block["widths"], repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), NAVY), ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
                    ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f5f7"), colors.white]),
                    ("LINEBELOW", (0,-1), (-1,-1), .5, colors.HexColor("#c9d5dd")),
                ]))
                story.extend([table, Spacer(1, 11)])
            elif kind == "image":
                story.append(Image(str(BASE / block["path"]), width=block["width"], height=block["height"]))
                story.append(Spacer(1, 8))
    doc = SimpleDocTemplate(str(BASE / "FINAL_REPORT.pdf"), pagesize=A4,
                            leftMargin=42, rightMargin=42, topMargin=42, bottomMargin=55,
                            title="Reciprocity and return: completed experiments", author="EMM Project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(BASE / "FINAL_REPORT.pdf")


if __name__ == "__main__":
    main()
