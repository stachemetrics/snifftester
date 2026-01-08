import gradio as gr

def hello(text):
    """Placeholder function - will become snifftest later."""
    return f"Hello! You entered: {text}"

def create_demo():
    """Create and return the Gradio demo."""
    demo = gr.Interface(
        fn=hello,
        inputs=gr.Textbox(label="Enter some text"),
        outputs=gr.Textbox(label="Output"),
        title="Snifftest - Hello World",
        description="Testing Modal deployment"
    )
    demo.queue()
    return demo

# For local testing: python app.py
if __name__ == "__main__":
    demo = create_demo()
    demo.launch()