import modal

app = modal.App("snifftest")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]",
        "gradio",
        "google-genai",
        "requests",
    )
    .add_local_file("app.py", "/root/app.py")
)

@app.function(
    image=image,
    max_containers=1,
    secrets=[modal.Secret.from_name("gemini-secret")],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Serve the Gradio app."""
    import sys
    sys.path.insert(0, "/root")
    
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from app import create_app

    demo = create_app()
    
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")