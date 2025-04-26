from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Function to create a PDF report for blood group prediction
def create_pdf(result, filename='prediction_report.pdf'):
    """
    Create a PDF report with the prediction details.
    
    :param result: Dictionary with 'blood_group' and 'confidence'
    :param filename: Output PDF filename
    """
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    # Title
    c.drawString(100, 750, "Blood Group Prediction Report")
    
    # Prediction Details
    c.drawString(100, 730, f"Predicted Blood Group: {result['blood_group']}")
    c.drawString(100, 710, f"Prediction Confidence: {result['confidence']:.2f}%")
    
    # Save PDF
    c.save()
    print(f"✅ PDF Report saved as {filename}")

# Example usage (to be called dynamically)
if __name__ == "__main__":
    # Example result - you should replace this with dynamic prediction results
    prediction_result = {
        "blood_group": "A+",   # Dynamically generated prediction
        "confidence": 98.75    # Dynamically generated confidence
    }
    create_pdf(prediction_result, "A_plus_prediction_report.pdf")
