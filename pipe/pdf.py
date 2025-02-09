from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import utils
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from io import BytesIO
import os
import argparse
def create_pdf_with_rescaled_pair(folder_path, output_pdf):
    # Get the paths for the specific text and image files



    # Create a PDF
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)

    # Initialize a list to store flowables
    story = []

    # Add the filename to the story
    filename_style = ParagraphStyle('Normal', spaceAfter=12)
    # filename_content = f"Filename: {base_filename}"
    filename_flowable = Paragraph(folder_path, filename_style)
    story.append(filename_flowable)
    unable_count = 0
    total_count = 0
    errordict = {}
    
    for image_file in os.listdir(folder_path):
        if not image_file.lower().endswith('.jpg'): continue
        # Calculate the width and height of the image (optional)
        # print(image_file)
        base_filename = image_file[:-4]
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
        text_files.sort()
        # Iterate through text files and add them to the story
        for text_file in text_files:
            # Extract the suffix of the text filename
            text_suffix = os.path.splitext(text_file)[0].split('_')[-1]

            text_path = os.path.join(folder_path, text_file)
            with open(text_path, 'r') as f:
                text_content = f.read()
            total_count+= 1

            # Replace newline characters with HTML line break tags
            text_content = text_content.replace('\n', '<br/>')

            # Create a style for the text
            styles = getSampleStyleSheet()
            text_style = ParagraphStyle('Normal', parent=styles['Normal'], spaceAfter=12)

            # Add the suffix to the content
            text_content_with_suffix = f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<({text_suffix}): {text_content}<br/>"
            if 'unable' in text_content:
                unable_count+= 1
                if text_suffix not in errordict:
                    errordict[text_suffix] = 0
                errordict[text_suffix]+= 1


            # Create a flowable for the auto-wrapped text
            text_flowable = Paragraph(text_content_with_suffix, text_style)
            story.append(text_flowable)

        # Build the PDF
    pdf.build(story)

    # Move the buffer cursor to the beginning
    buffer.seek(0)

    # Write the buffer content to the output PDF file
    print(f'Failed to answer: {unable_count}/{total_count} ({unable_count/total_count:.2%})')
    for key in sorted(errordict.keys()):
        print(f"{key}: {errordict[key]}")
    with open(output_pdf, 'wb') as output_file:
        output_file.write(buffer.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Path to the folder containing the images", default="img")
    args = parser.parse_args()

    folder_path = args.folder
    output_pdf = "output.pdf"
    create_pdf_with_rescaled_pair(folder_path, f"{folder_path}/{output_pdf}")
