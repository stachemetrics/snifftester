import os
import json
import gradio as gr
from google import genai
from google.genai import types

# Initialize client globally
client = None

def get_client():
    global client
    if client is None:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client

# ============== PROMPTS ==============

EXTRACTION_PROMPT = """
You are a reference extraction system. Extract ALL references from this AI-generated text.

Only extract information EXPLICITLY stated. Use null for anything not directly written.

For each reference provide:
- raw_text: Exact citation as written
- url: URL if present, null otherwise
- source_type: One of: academic_journal, academic_preprint, news_media, government, social_media, personal_blog, documentation, commercial, unknown
- platform_name: Specific platform/publication name

TEXT TO ANALYZE:
---
{text}
---

Return ONLY a JSON array. Do not hallucinate - null for anything not explicit.
"""

CRAAP_PROMPT = """
You are a SKEPTICAL source verification system. Evaluate this reference using web search.

REFERENCE:
{reference_json}

Search for this source and evaluate STRICTLY on CRAAP:

**Currency (1-5):** How recent? Still relevant?
**Relevance (1-5):** Is this a substantive source for the topic?
**Authority (1-5):** Who wrote it? Are they credible? Personal blogs/Substack START at 2.
**Accuracy (1-5):** Can claims be verified? Does it cite sources?
**Purpose (1-5):** Inform, persuade, or sell?

BE CRITICAL. Most sources are mediocre.

Return JSON only:
{{
  "currency": {{"score": 1-5, "evidence": "..."}},
  "relevance": {{"score": 1-5, "evidence": "..."}},
  "authority": {{"score": 1-5, "evidence": "..."}},
  "accuracy": {{"score": 1-5, "evidence": "..."}},
  "purpose": {{"score": 1-5, "evidence": "..."}},
  "overall_score": 1.0-5.0,
  "summary": "One sentence"
}}
"""

# ============== CORE FUNCTIONS ==============

def extract_references(text: str) -> list[dict]:
    """Extract references from text."""
    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=EXTRACTION_PROMPT.format(text=text)
    )
    
    try:
        clean = response.text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        return []

def verify_reference(reference: dict) -> dict:
    """Verify a single reference using CRAAP with Google Search."""
    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=CRAAP_PROMPT.format(reference_json=json.dumps(reference, indent=2)),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    
    try:
        clean = response.text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        return {"overall_score": None, "summary": "Failed to verify"}

def calculate_snifftest(verified_refs: list[dict]) -> dict:
    """Calculate overall Snifftest score."""
    scores = [v["craap"]["overall_score"] for v in verified_refs 
              if v.get("craap", {}).get("overall_score")]
    
    if not scores:
        return {"score": 0, "label": "Unknown", "emoji": "❓"}
    
    mean_score = sum(scores) / len(scores)
    min_score = min(scores)
    low_count = sum(1 for s in scores if s < 2.5)
    
    if min_score < 2 or low_count >= 2:
        label, emoji = "Foul", "🤢"
    elif mean_score < 2.5 or low_count >= 1:
        label, emoji = "Funky", "😬"
    elif mean_score < 3.5:
        label, emoji = "Fresh", "😊"
    else:
        label, emoji = "Sweet", "🌟"
    
    return {
        "score": round(mean_score, 2),
        "label": label,
        "emoji": emoji,
        "min_score": min_score,
        "num_refs": len(scores),
        "low_count": low_count
    }

# ============== MAIN PIPELINE ==============

def run_snifftest(text: str, progress=gr.Progress()):
    """Run the full Snifftest pipeline with progress updates."""
    
    if not text.strip():
        yield "👃 Please paste some AI-generated text to sniff!", ""
        return
    
    # Stage 1: Extract
    progress(0, desc="👃 Sniffing for references...")
    yield "👃 Sniffing for references...", ""
    
    refs = extract_references(text)
    
    if not refs:
        yield "👃 No references found to sniff!", ""
        return
    
    yield f"👃 Found {len(refs)} references. Starting verification...", ""
    
    # Stage 2: Verify each reference
    verified = []
    for i, ref in enumerate(refs):
        name = ref.get("platform_name") or ref.get("url", "unknown")[:30]
        progress((i + 1) / len(refs), desc=f"🔍 Checking {name}...")
        yield f"🔍 Verifying reference {i+1}/{len(refs)}: {name}", ""
        
        craap = verify_reference(ref)
        verified.append({"reference": ref, "craap": craap})
    
    # Stage 3: Calculate final score
    progress(1, desc="📊 Calculating Snifftest...")
    snifftest = calculate_snifftest(verified)
    
    # Build output
    header = f"""
# {snifftest['emoji']} SNIFFTEST: {snifftest['label'].upper()} ({snifftest['score']}/5)

**{snifftest['num_refs']} sources analyzed** | Lowest: {snifftest['min_score']}/5 | Low quality: {snifftest['low_count']}

---
"""
    
    details = "## 📚 Reference Details\n\n"
    for v in verified:
        ref = v["reference"]
        craap = v["craap"]
        name = ref.get("platform_name") or ref.get("url", "unknown")[:40]
        score = craap.get("overall_score", "?")
        summary = craap.get("summary", "No summary")
        
        # Score emoji
        if score and score >= 4:
            score_emoji = "✅"
        elif score and score >= 2.5:
            score_emoji = "⚠️"
        else:
            score_emoji = "❌"
        
        details += f"### {score_emoji} {name} — {score}/5\n"
        details += f"*{ref.get('source_type', 'unknown')}*\n\n"
        details += f"{summary}\n\n"
        
        if craap.get("currency"):
            details += f"- **C**urrency: {craap['currency']['score']}/5\n"
            details += f"- **R**elevance: {craap['relevance']['score']}/5\n"
            details += f"- **A**uthority: {craap['authority']['score']}/5\n"
            details += f"- **A**ccuracy: {craap['accuracy']['score']}/5\n"
            details += f"- **P**urpose: {craap['purpose']['score']}/5\n\n"
        
        details += "---\n\n"
    
    yield header, details

# ============== GRADIO APP ==============

def create_app():
    """Create the Snifftest Gradio app."""
    
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
    )
    
    with gr.Blocks(theme=theme, title="👃 Snifftest") as demo:
        gr.Markdown("""
        # 👃 Snifftest
        ### Does this AI response pass the smell test?
        
        Paste any AI-generated text below. Snifftest will extract the references 
        and evaluate each one using the **CRAAP framework** (Currency, Relevance, 
        Authority, Accuracy, Purpose).
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="📝 Paste AI-generated text",
                    placeholder="Paste the AI response you want to fact-check...",
                    lines=12,
                )
                submit_btn = gr.Button("👃 Sniff It!", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                status = gr.Markdown(label="Status", value="*Waiting for input...*")
                result = gr.Markdown(label="Results")
        
        submit_btn.click(
            fn=run_snifftest,
            inputs=[input_text],
            outputs=[status, result],
        )
        
        gr.Markdown("""
        ---
        **Snifftest Ratings:**
        🌟 Sweet (4+) | 😊 Fresh (3-4) | 😬 Funky (2.5-3) | 🤢 Foul (<2.5)
        """)
    
    return demo