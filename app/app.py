import os
import gradio as gr
from google import genai

def create_app():
    """Create and return the Gradio blocks."""
    
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def analyze(text):
        if not text.strip():
            return "Please enter some text to analyze."
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Summarize this in one sentence: {text}"
        )
        return response.text

    demo = gr.Interface(
        fn=analyze,
        inputs=gr.Textbox(label="Enter text", lines=5),
        outputs=gr.Textbox(label="Gemini response"),
        title="Snifftest - Gemini Test",
        description="Testing Gemini API integration"
    )

    return demo

# For local testing: python app.py
if __name__ == "__main__":
    demo = create_demo()
    demo.launch()