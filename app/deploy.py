import modal

app = modal.App("snifftest")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]",
    "gradio",
    "google-genai",
    "requests",
)

with image.imports():
    import os
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from fastapi import FastAPI
    from google import genai

@app.function(
    image=image,
    max_containers=1,
    secrets=[modal.Secret.from_name("gemini-secret")],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Snifftest Gradio app."""
    
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

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")