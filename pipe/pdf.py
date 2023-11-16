from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import utils
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from io import BytesIO
import os

def create_pdf_with_rescaled_pair(folder_path, output_pdf, base_filename):
    # Get the paths for the specific text and image files
    text_file = f"{base_filename}.txt"
    image_file = f"{base_filename}.jpg"

    # Check if the files exist
    if not os.path.exists(os.path.join(folder_path, text_file)) or not os.path.exists(os.path.join(folder_path, image_file)):
        print(f"Files {text_file} or {image_file} not found.")
        return

    # Create a PDF
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)

    # Calculate the width and height of the image (optional)
    img_path = os.path.join(folder_path, image_file)
    img = utils.ImageReader(img_path)
    img_width, img_height = img.getSize()

    # Calculate the scaling factor to fit the image into 1/4 of the page
    scale_factor = 0.25 / max(img_width / 72, img_height / 72)

    # Rescale the image
    img_width *= scale_factor
    img_height *= scale_factor

    # Create a flowable for the image
    img_flowable = utils.Image(img_path, width=img_width, height=img_height)

    # Auto-wrap text below the image
    text_path = os.path.join(folder_path, text_file)
    with open(text_path, 'r') as f:
        text_content = f.read()

    # Create a style for the text
    styles = getSampleStyleSheet()
    text_style = ParagraphStyle('Normal', parent=styles['Normal'], spaceAfter=12)

    # Create a flowable for the auto-wrapped text
    text_flowable = Paragraph(text_content, text_style)

    # Build the story with image and text
    story = [img_flowable, text_flowable]

    # Build the PDF
    pdf.build(story)

    # Move the buffer cursor to the beginning
    buffer.seek(0)

    # Write the buffer content to the output PDF file
    with open(output_pdf, 'wb') as output_file:
        output_file.write(buffer.read())

if __name__ == "__main__":
    folder_path = "img"
    output_pdf = "output.pdf"
    base_filename = "002"
    create_pdf_with_rescaled_pair(folder_path, output_pdf, base_filename)
