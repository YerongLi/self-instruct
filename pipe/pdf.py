from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import utils
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from io import BytesIO
import os

def create_pdf_with_rescaled_pair(folder_path, output_pdf, base_filename):
    # Get the paths for the specific text and image files
    image_file = f"{base_filename}.jpg"

    # Check if the image file exists
    if not os.path.exists(os.path.join(folder_path, image_file)):
        print(f"Image file {image_file} not found.")
        return

    # Create a PDF
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)

    # Initialize a list to store flowables
    story = []

    # Add the filename to the story
    filename_style = ParagraphStyle('Normal', spaceAfter=12)
    filename_content = f"Filename: {base_filename}"
    filename_flowable = Paragraph(filename_content, filename_style)
    story.append(filename_flowable)

    # Calculate the width and height of the image (optional)
    img_path = os.path.join(folder_path, image_file)
    img = utils.ImageReader(img_path)
    img_width, img_height = img.getSize()

    # Set a fixed width for the image
    fixed_width = 200
    scale_factor = fixed_width / img_width

    # Rescale the image
    img_width *= scale_factor
    img_height *= scale_factor

    # Create a flowable for the image
    img_flowable = Image(img_path, width=fixed_width, height=img_height)
    story.append(img_flowable)

    # Find all text files with the same prefix
    text_files = [file for file in os.listdir(folder_path) if file.startswith(base_filename + '_') and file.lower().endswith('.txt')]

    # Iterate through text files and add them to the story
    for text_file in text_files:
        # Extract the suffix of the text filename
        text_suffix = os.path.splitext(text_file)[0].split('_')[-1]

        text_path = os.path.join(folder_path, text_file)
        with open(text_path, 'r') as f:
            text_content = f.read()

        # Replace newline characters with HTML line break tags
        text_content = text_content.replace('\n', '<br/>')

        # Add the suffix to the content
        text_content_with_suffix = f"Text ({text_suffix}): {text_content}"

        # Create a flowable for the auto-wrapped text
        text_flowable = Paragraph(text_content_with_suffix, text_style)
        story.append(text_flowable)

        # Add a line break after the centered text
        story.append(Spacer(1, 12))  # Adjust the vertical space as needed

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
