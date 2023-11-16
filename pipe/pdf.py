from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import utils
import os

def create_pdf_with_rescaled_pair(folder_path, output_pdf, base_filename):
    # Get the paths for the specific text and image files
    text_file = f"{base_filename}.txt"
    image_file = f"{base_filename}.jpg"
    print(image_file)
    # Check if the files exist
    if not os.path.exists(os.path.join(folder_path, text_file)) or not os.path.exists(os.path.join(folder_path, image_file)):
        print(f"Files {text_file} or {image_file} not found.")
        return

    # Create a PDF
    c = canvas.Canvas(output_pdf, pagesize=letter)

    # Calculate the width and height of the image (optional)
    img_path = os.path.join(folder_path, image_file)
    img = utils.ImageReader(img_path)
    img_width, img_height = img.getSize()

    # Calculate the scaling factor to fit the image into 1/4 of the page
    scale_factor = 0.25 / max(img_width / 72, img_height / 72)

    # Rescale the image
    img_width *= scale_factor
    img_height *= scale_factor

    # Draw text
    text_path = os.path.join(folder_path, text_file)
    with open(text_path, 'r') as f:
        text_content = f.read()
    c.drawString(100, 700, text_content)

    # Draw rescaled image
    c.drawInlineImage(img_path, 100, 500, width=img_width, height=img_height)

    # Save the PDF
    c.save()

if __name__ == "__main__":
    folder_path = "img"
    output_pdf = "output.pdf"
    base_filename = "002"
    create_pdf_with_rescaled_pair(folder_path, output_pdf, base_filename)
