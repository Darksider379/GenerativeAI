import PyPDF2
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models


def extract_text_from_pdf(pdf_path):
  """
  Extracts text from a PDF file using PyPDF2.

  Args:
      pdf_path (str): Path to the PDF file.

  Returns:
      str: Extracted text content from the PDF.
  """

  with open(pdf_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

# Example usage
pdf_path = "Add_your_absolute_filepath_here"  # Replace with your PDF path
text = extract_text_from_pdf(pdf_path)

# print(text)

def generate():
  vertexai.init(project="gcp_project_id", location="us-central1")
  model = GenerativeModel("gemini-1.0-pro-vision-001")
  responses = model.generate_content(f"""
  Extract the skills from the below snippet:
  {text}
  """,
    generation_config={
      "max_output_tokens": 2048,
      "temperature": 0.4,
      "top_p": 1,
      "top_k": 32
    },
    safety_settings={
      generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    stream=True,
  )

  for response in responses:
    print(response.text, end="")


generate()
