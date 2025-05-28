import re # Import regex for Markdown conversion

# ReportLab Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, BaseDocTemplate, Frame, PageTemplate, Preformatted, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.units import inch, cm
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from html import escape
import traceback
import pandas as pd
import plotly.graph_objects as go
import io

# --- Helper Function to Convert Results to ReportLab Flowables ---
# This function is NO LONGER CALLED by the simplified generate_pdf_reportlab,
# but kept here in case it's used elsewhere or for future reference.
def result_to_flowables(result, styles):
    """Converts various result types into a list of ReportLab Flowables."""
    flowables = []
    spacer_small = Spacer(1, 0.1*inch)
    # ... (rest of the function remains the same as before) ...
    if isinstance(result, pd.DataFrame):
        try:
            df_display = result
            data = [df_display.columns.to_list()] + df_display.values.tolist()
            safe_data = [[escape(str(item)) for item in row] for row in data]
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            table = Table(safe_data, style=style, hAlign='LEFT', repeatRows=1)
            flowables.append(table)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error converting DataFrame to Table: {e}\n{tb_str}")
            flowables.append(Paragraph(f"<i>Error displaying DataFrame: {escape(str(e))}</i>", styles['Italic']))

    elif isinstance(result, go.Figure):
        try:
            img_bytes = result.to_image(format='png', scale=2)
            img_file = io.BytesIO(img_bytes)
            img_reader = ImageReader(img_file)
            img_width, img_height = img_reader.getSize()
            max_width = 6.5 * inch
            aspect = img_height / float(img_width)
            display_width = min(img_width, max_width)
            display_height = display_width * aspect
            img = Image(img_file, width=display_width, height=display_height)
            img.hAlign = 'CENTER'
            flowables.append(img)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error converting Plotly figure to Image: {e}\n{tb_str}")
            flowables.append(Paragraph(f"<i>Error rendering Plotly figure: {escape(str(e))}</i>", styles['Italic']))

    elif isinstance(result, list):
        for i, item in enumerate(result):
            item_flowables = result_to_flowables(item, styles)
            if item_flowables:
                flowables.extend(item_flowables)
                if i < len(result) - 1:
                    flowables.append(Spacer(1, 0.1*inch)) # Use explicit spacer value

    elif isinstance(result, (str, int, float, bool)):
        safe_text = escape(str(result))
        flowables.append(Preformatted(safe_text, styles['Code']))

    elif result is None:
        flowables.append(Paragraph("<i>No output generated for this step.</i>", styles['Italic']))
    else:
        safe_text = escape(str(result))
        flowables.append(Paragraph(f"<i>Unrenderable result type: {escape(type(result).__name__)}</i>", styles['Italic']))
        flowables.append(Preformatted(safe_text, styles['Code']))

    return flowables

# --- Helper for Markdown to ReportLab HTML Subset ---
def basic_markdown_to_rl_html(md_text):
    """Converts basic Markdown (bold, italic, bullets, H3/H4) to ReportLab HTML subset."""
    if not isinstance(md_text, str):
        md_text = str(md_text)

    # 1. Escape HTML-sensitive characters first
    text = escape(md_text)

    # --- CORRECTED: Handle H3 OR H4 Headings (### or #### text) ---
    # Changed regex to match 3 OR 4 hash characters: #{3,4}
    text = re.sub(r'^#{3,4}\s+(.*)', r'<br/><font size="+1"><b>\1</b></font>', text, flags=re.MULTILINE)
    # --- END CORRECTION ---

    # 2. Convert Markdown bold (**text**) to HTML bold (<b>text</b>)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    # 3. Convert Markdown italics (*text* or _text_) to HTML italics (<i>text</i>)
    text = re.sub(r'[\*_](.*?)[\*_]', r'<i>\1</i>', text)

    # 4. Convert Markdown-style list items (* item or - item) to bullet points
    text = re.sub(r'^\s*[\*\-]\s+(.*)', r'&bull; \1', text, flags=re.MULTILINE)

    # 5. Convert remaining newlines to <br/> tags
    text = text.replace('\n', '<br/>')

    # Remove potential leading <br/> added by H3/H4 if it's the very first line
    if text.startswith('<br/>'):
        text = text[len('<br/>'):]

    # Optional: Handle "Data Sources:" specifically?
    # text = re.sub(r'^(Data Sources:.*)', r'<b>\1</b>', text, flags=re.MULTILINE)

    return text

# --- Main PDF Generation Function (Simplified) ---

def generate_pdf_reportlab(filename, report):
    """Generates a simplified PDF report using ReportLab, containing
       only the title and the final report summary.
       Ignores user_input, plan, and plan_execution arguments.
    """
    # Although parameters user_input, plan, plan_execution are accepted,
    # they are ignored in this simplified version.
    print(f"Generating simplified PDF report: {filename}") # Log which version runs
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()

        story = []
        spacer_large = Spacer(1, 0.3*inch)
        spacer_medium = Spacer(1, 0.2*inch) # Keep for spacing after headings
        spacer_small = Spacer(1, 0.1*inch)  # Keep for spacing before report text

        # 1. Title
        story.append(Paragraph("Deep Insights Report", styles['Heading1']))
        story.append(spacer_large) # Space after title

        # --- USER INPUT SECTION REMOVED ---
        # --- GENERATED PLAN SECTION REMOVED ---
        # --- PLAN EXECUTION & RESULTS SECTION REMOVED ---

        # 2. Final Report Summary (Now appears directly after title)
        # No page break needed here anymore
        story.append(Paragraph("Final Report Summary", styles['Heading2']))
        story.append(spacer_small) # Small space before the report body
        if report:
            # Convert basic Markdown to ReportLab HTML subset
            processed_report_html = basic_markdown_to_rl_html(report)
            # Create Paragraph using the processed HTML
            story.append(Paragraph(processed_report_html, styles['Normal']))
        else:
            story.append(Paragraph("<i>Report summary was not provided.</i>", styles['Italic']))

        # Build the PDF
        doc.build(story)
        print(f"Report successfully generated: {filename}")
        return True # Indicate success

    except Exception as e:
        # Use Streamlit components only if this function is exclusively run in a Streamlit context
        # Otherwise, rely on printing for error logs
        # st.error(f"Error generating PDF using ReportLab: {e}")
        tb_str = traceback.format_exc()
        # st.code(tb_str, language='text')
        print(f"Error generating PDF: {e}\n{tb_str}") # Print error to console/log
        return False # Indicate failure
