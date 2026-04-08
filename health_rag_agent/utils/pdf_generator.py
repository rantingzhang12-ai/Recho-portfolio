# utils/pdf_generator.py

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(content: str, filename="health_report.pdf"):
    """
    生成PDF报告
    """
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(content, styles["Normal"]))

    doc.build(story)

    return filename
