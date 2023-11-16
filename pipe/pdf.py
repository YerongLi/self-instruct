from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import utils
import os

def create_pdf_with_pairs(folder_path, output_pdf):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter text and image files
    text_files = [file for file in files if file.lower().endswith('.txt')]
    image_files = [file for file in files if file.lower().endswith('.jpg')]

    # Sort the files to ensure proper pairing
    text_files.sort()
    image_files.sort()

    # Create a PDF
    c = canvas.Canvas(output_pdf, pagesize=letter)

    for text_file, image_file in zip(text_files, image_files):
        # Calculate the width and height of the image (optional)
        img_path = os.path.join(folder_path, image_file)
        img = utils.ImageReader(img_path)
        img_width, img_height = img.getSize()

        # Draw text
        text_path = os.path.join(folder_path, text_file)
        with open(text_path, 'r') as f:
            text_content = f.read()
        c.drawString(100, 700, text_content)

        # Draw image
        c.drawInlineImage(img_path, 100, 500, width=img_width, height=img_height)

        # Add a new page for the next pair
        c.showPage()

    c.save()

if __name__ == "__main__":
    folder_path = "img"
    output_pdf = "output.pdf"
    create_pdf_with_pairs(folder_path, output_pdf)
