from __future__ import annotations

from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(
    record: dict,
    original_path: str | Path,
    highlighted_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Generate a professional diagnostic PDF report for an MRI scan.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Page setup
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles matching our primary theme (#0f766e)
    primary_color = colors.HexColor("#0f766e")
    text_dark = colors.HexColor("#1e293b")
    
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=primary_color,
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15,
    )
    
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6,
        borderPadding=(0, 0, 2, 0),
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        textColor=text_dark,
        leading=14,
    )
    
    bold_body_style = ParagraphStyle(
        "BoldBodyDark",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=colors.HexColor("#0f766e"),
        leading=10,
    )
    
    story = []
    
    # 1. Header Section
    story.append(Paragraph("Brain Tumor Classification System", title_style))
    story.append(Paragraph("AI-Assisted Diagnostic Triage & Decision Support Report", subtitle_style))
    
    # 2. Patient & Scan Metadata Table
    patient_data = [
        [
            Paragraph("<b>Patient ID:</b>", body_style),
            Paragraph(str(record.get("patient_id", "N/A")), body_style),
            Paragraph("<b>Scan Timestamp:</b>", body_style),
            Paragraph(str(record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))), body_style),
        ],
        [
            Paragraph("<b>Age:</b>", body_style),
            Paragraph(f"{record.get('patient_age', 'N/A')} yrs", body_style),
            Paragraph("<b>Gender:</b>", body_style),
            Paragraph(str(record.get("patient_gender", "N/A")).capitalize(), body_style),
        ]
    ]
    
    meta_table = Table(patient_data, colWidths=[100, 160, 110, 160])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # 3. Prediction Results Section
    story.append(Paragraph("Analysis & AI Prediction", section_heading))
    
    # Format confidence level
    conf = record.get("confidence", 0.0)
    risk_interpretation = "Low Confidence"
    if conf >= 0.9:
        risk_interpretation = "High Confidence (Strong Support)"
    elif conf >= 0.7:
        risk_interpretation = "Moderate Confidence (Screening Level)"
        
    prediction_data = [
        [
            Paragraph("<b>Predicted Classification</b>", bold_body_style),
            Paragraph("<b>Model Confidence Score</b>", bold_body_style),
            Paragraph("<b>Reliability Interpretation</b>", bold_body_style)
        ],
        [
            Paragraph(f"<font color='#0f766e'><b>{record.get('prediction_class', 'N/A')}</b></font>", bold_body_style),
            Paragraph(f"{conf:.2%}", body_style),
            Paragraph(risk_interpretation, body_style)
        ]
    ]
    pred_table = Table(prediction_data, colWidths=[200, 150, 180])
    pred_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecfeff")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#a5f3fc")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.append(pred_table)
    story.append(Spacer(1, 15))
    
    # 4. Image Visualizations
    story.append(Paragraph("MRI Scan Visualizations & Explainability (Grad-CAM)", section_heading))
    
    image_elements = []
    # If paths are provided and exist, embed them
    original_exists = Path(original_path).exists()
    highlighted_exists = Path(highlighted_path).exists()
    
    if original_exists and highlighted_exists:
        # Load and resize images to fit nicely side by side (240x240 pt)
        img_orig = Image(str(original_path), width=240, height=240)
        img_grad = Image(str(highlighted_path), width=240, height=240)
        
        img_table_data = [
            [img_orig, img_grad],
            [Paragraph("<font size='8'>Original Input MRI Scan</font>", subtitle_style), 
             Paragraph("<font size='8'>Grad-CAM Convolutional Layer Highlights</font>", subtitle_style)]
        ]
        img_table = Table(img_table_data, colWidths=[265, 265])
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        story.append(img_table)
    else:
        story.append(Paragraph("<i>Visual scan images could not be loaded into the PDF document.</i>", body_style))
        
    story.append(Spacer(1, 10))
    
    # 5. Clinician/Physician Notes Section
    story.append(Paragraph("Clinician Assessment & Action Plan", section_heading))
    notes_text = record.get("physician_notes") or "No clinician notes provided at the time of report generation."
    
    notes_table = Table([[Paragraph(notes_text, body_style)]], colWidths=[530])
    notes_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ]
        )
    )
    story.append(notes_table)
    story.append(Spacer(1, 15))
    
    # 6. Sign-off / Signature Area
    sig_data = [
        [
            Paragraph("<b>Reporting Clinician:</b> ___________________________", body_style),
            Paragraph("<b>Signature:</b> ___________________________", body_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[265, 265])
    sig_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 15),
            ]
        )
    )
    story.append(sig_table)
    story.append(Spacer(1, 15))
    
    # 7. Medical Disclaimer
    story.append(Paragraph("<b>Disclaimer:</b> This AI-assisted classification report is generated by an automated convolutional neural network trained on open-source datasets. It is designed solely for research, validation, and educational decision support. It does NOT constitute a clinical diagnosis or treatment recommendation. Final diagnosis must always be made by a qualified healthcare professional.", disclaimer_style))
    
    # Build Document
    doc.build(story)
    return output_path
