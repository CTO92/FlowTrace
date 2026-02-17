import os
from fpdf import FPDF
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'FlowTrace - High Conviction Signal Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_signal_report(df_signals):
    """
    Generates a PDF report for the provided signals DataFrame.
    Returns the binary content of the PDF.
    """
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Report Metadata
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"Total Signals: {len(df_signals)}", 0, 1)
    pdf.ln(5)
    
    if df_signals.empty:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, "No high-conviction signals found matching criteria.", 0, 1)
        return pdf.output(dest='S').encode('latin-1')

    for index, row in df_signals.iterrows():
        # Signal Header
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(200, 220, 255)
        title = f"{row['source_ticker']} -> {row['target_ticker']} ({row['event_type']})"
        pdf.cell(0, 10, title, 0, 1, 'L', fill=True)
        
        # Metrics
        pdf.set_font("Arial", size=10)
        metrics = f"Confidence: {row['confidence']}% | Unified Score: {row['unified_score']} | Exp Move: {row['expected_move_pct']}%"
        pdf.cell(0, 8, metrics, 0, 1)
        
        # Summary
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 6, "Summary:", 0, 1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, str(row['summary']))
        pdf.ln(2)
        
        # Reasoning
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 6, "Reasoning:", 0, 1)
        pdf.set_font("Arial", size=10)
        # Clean up reasoning text slightly if needed
        reasoning = str(row['reasoning']).replace("\n", " ")
        pdf.multi_cell(0, 5, reasoning)
        
        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')