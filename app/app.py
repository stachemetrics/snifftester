import os
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
import gradio as gr
from google import genai
from google.genai import types

# ============== LOGGING ==============

LOG_PATH = "/root/logs/snifftest_logs.jsonl"

@dataclass
class LLMLog:
    timestamp: str
    session_id: str
    model: str
    task: str
    prompt_template: str
    prompt_content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    
    def to_dict(self):
        return asdict(self)

def append_log(log: LLMLog, volume=None):
    """Append log entry to persistent storage."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")
        if volume:
            volume.commit()
    except Exception as e:
        print(f"Logging error: {e}")

# ============== CLIENT ==============

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
- stated_author: Author name if explicitly written, null otherwise
- stated_date: Publication date/year if written, null otherwise

TEXT TO ANALYZE:
---
{text}
---

Return ONLY a JSON array. Do not hallucinate - null for anything not explicit.
"""

CRAAP_PROMPT = """
You are a SKEPTICAL source verification system. Your job is to be critical and identify weak sources.

REFERENCE:
{reference_json}

Search for this source and evaluate STRICTLY on CRAAP. Be harsh - most sources are mediocre.

SCORING GUIDE (err on the lower side):

**Currency (1-5):**
- 5: Published within last year, regularly updated
- 3: 1-3 years old, still somewhat relevant  
- 1: Outdated, stale, or no date available

**Relevance (1-5):**
- 5: Primary source, directly addresses topic with depth
- 3: Tangentially related, surface-level coverage
- 1: Barely relevant, clickbait title, or off-topic

**Authority (1-5):**
- 5: Named expert with verifiable credentials, institutional backing, peer-reviewed
- 3: Professional journalist at known outlet, or practitioner with some track record
- 2: Anonymous or pseudonymous author, no credentials stated
- 1: Random blog, no author info, self-published with no reputation

IMPORTANT: Personal blogs, Substack, Medium, etc. should START at 2 and only go higher if the author has VERIFIABLE expertise (real name, credentials, institutional affiliation). "Thought leaders" and "consultants" without specific credentials = 2.

**Accuracy (1-5):**
- 5: Cites primary sources, data is verifiable, peer-reviewed
- 3: Makes claims with some supporting links, but not rigorous
- 1: No citations, unverifiable claims, or contradicted by reliable sources

**Purpose (1-5):**
- 5: Educational, informational, no commercial motive
- 3: Some bias but primarily informative
- 1: Selling something, affiliate links, rage-bait, or propaganda

Also check: Is the URL accessible?

BE CRITICAL. A typical personal blog with no credentials should score 2-2.5 overall. Only authoritative, well-sourced content deserves 4+.

Return JSON only:
{{
  "url_accessible": true/false/null,
  "currency": {{"score": 1-5, "evidence": "..."}},
  "relevance": {{"score": 1-5, "evidence": "..."}},
  "authority": {{"score": 1-5, "evidence": "..."}},
  "accuracy": {{"score": 1-5, "evidence": "..."}},
  "purpose": {{"score": 1-5, "evidence": "..."}},
  "overall_score": 1.0-5.0,
  "red_flags": ["list", "any", "concerns"],
  "summary": "One critical sentence"
}}
"""

SNIFFTEST_SUMMARY_PROMPT = """
You are the Snifftest summarizer. Given CRAAP evaluations of all sources in an AI-generated response, write a blunt, honest summary.

SNIFFTEST RESULT: {label} {emoji} ({score}/5)

VERIFIED SOURCES:
{verified_json}

Write a 2-3 sentence summary that:
1. States the overall verdict clearly
2. Calls out the weakest sources specifically
3. Notes any red flags
4. Suggests what would improve the response (if applicable)

Be direct and a little snarky. Examples:
- "This response smells funky. Half the sources are random blogs with no credentials..."
- "Pretty sweet! All sources are peer-reviewed or from reputable institutions..."
- "Foul. The main claim relies entirely on a 2019 Medium post by someone called 'CryptoGuru'..."

Keep it under 50 words. No JSON, just plain text.
"""

# ============== CORE FUNCTIONS ==============

def extract_references(text: str, session_id: str, volume=None) -> tuple[list[dict], LLMLog]:
    """Extract references from text."""
    start_time = time.time()
    
    prompt_content = EXTRACTION_PROMPT.format(text=text)
    
    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt_content
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    usage = response.usage_metadata
    
    log = LLMLog(
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
        model="gemini-2.0-flash",
        task="extract_references",
        prompt_template="EXTRACTION_PROMPT",
        prompt_content=prompt_content,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        total_tokens=usage.total_token_count,
        latency_ms=latency_ms
    )
    append_log(log, volume)
    
    try:
        clean = response.text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip()), log
    except json.JSONDecodeError:
        return [], log

def verify_reference(reference: dict, session_id: str, volume=None) -> tuple[dict, LLMLog]:
    """Verify a single reference using CRAAP with Google Search."""
    start_time = time.time()
    
    prompt_content = CRAAP_PROMPT.format(reference_json=json.dumps(reference, indent=2))
    
    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt_content,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    usage = response.usage_metadata
    
    log = LLMLog(
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
        model="gemini-2.0-flash",
        task="verify_reference",
        prompt_template="CRAAP_PROMPT",
        prompt_content=prompt_content,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        total_tokens=usage.total_token_count,
        latency_ms=latency_ms
    )
    append_log(log, volume)
    
    try:
        clean = response.text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip()), log
    except json.JSONDecodeError:
        return {"overall_score": None, "summary": "Failed to verify", "red_flags": []}, log

def generate_snarky_summary(verified_refs: list[dict], snifftest: dict, session_id: str, volume=None) -> tuple[str, LLMLog]:
    """Generate a snarky summary of the Snifftest results."""
    start_time = time.time()
    
    prompt_content = SNIFFTEST_SUMMARY_PROMPT.format(
        label=snifftest["label"],
        emoji=snifftest["emoji"],
        score=snifftest["score"],
        verified_json=json.dumps(verified_refs, indent=2)
    )
    
    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt_content
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    usage = response.usage_metadata
    
    log = LLMLog(
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
        model="gemini-2.0-flash",
        task="generate_summary",
        prompt_template="SNIFFTEST_SUMMARY_PROMPT",
        prompt_content=prompt_content,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        total_tokens=usage.total_token_count,
        latency_ms=latency_ms
    )
    append_log(log, volume)
    
    return response.text.strip(), log

def calculate_snifftest(verified_refs: list[dict]) -> dict:
    """Calculate overall Snifftest score."""
    scores = [v["craap"]["overall_score"] for v in verified_refs 
              if v.get("craap", {}).get("overall_score")]
    
    red_flags = []
    for v in verified_refs:
        if v.get("craap", {}).get("red_flags"):
            red_flags.extend(v["craap"]["red_flags"])
    
    if not scores:
        return {"score": 0, "label": "Unknown", "emoji": "❓", "red_flags": red_flags}
    
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
        "low_count": low_count,
        "red_flags": list(set(red_flags))
    }

# ============== MAIN PIPELINE ==============

def run_snifftest(text: str, volume=None, progress=gr.Progress()):
    """Run the full Snifftest pipeline with progress updates."""
    
    # Generate unique session ID for this run
    session_id = str(uuid.uuid4())[:8]
    
    if not text.strip():
        yield "👃 Please paste some AI-generated text to sniff!", ""
        return
    
    # Stage 1: Extract
    progress(0, desc="👃 Sniffing for references...")
    yield f"👃 Sniffing for references... (session: {session_id})", ""
    
    refs, extract_log = extract_references(text, session_id, volume)
    
    if not refs:
        yield "👃 No references found to sniff!", ""
        return
    
    yield f"👃 Found {len(refs)} references ({extract_log.latency_ms}ms). Starting verification...", ""
    
    # Stage 2: Verify each reference
    verified = []
    for i, ref in enumerate(refs):
        name = ref.get("platform_name") or ref.get("url", "unknown")[:30]
        progress((i + 0.5) / (len(refs) + 1), desc=f"🔍 Checking {name}...")
        yield f"🔍 Verifying reference {i+1}/{len(refs)}: {name}", ""
        
        craap, verify_log = verify_reference(ref, session_id, volume)
        verified.append({"reference": ref, "craap": craap})
        
        # Show intermediate score
        score = craap.get("overall_score", "?")
        yield f"🔍 Verified {i+1}/{len(refs)}: {name} → {score}/5", ""
    
    # Stage 3: Calculate final score
    progress(len(refs) / (len(refs) + 1), desc="📊 Calculating Snifftest...")
    yield "📊 Calculating Snifftest score...", ""
    snifftest = calculate_snifftest(verified)
    
    # Stage 4: Generate snarky summary
    progress((len(refs) + 0.5) / (len(refs) + 1), desc="✍️ Writing summary...")
    yield "✍️ Writing snarky summary...", ""
    snarky_summary, summary_log = generate_snarky_summary(verified, snifftest, session_id, volume)
    
    progress(1, desc="✅ Done!")
    
    # Build output
    header = f"""
# {snifftest['emoji']} SNIFFTEST: {snifftest['label'].upper()} ({snifftest['score']}/5)

> {snarky_summary}

**{snifftest['num_refs']} sources analyzed** | Lowest: {snifftest['min_score']}/5 | Low quality: {snifftest['low_count']} | Session: `{session_id}`
"""
    
    if snifftest.get("red_flags"):
        header += f"\n🚩 **Red flags:** {', '.join(snifftest['red_flags'][:5])}\n"
    
    header += "\n---\n"
    
    details = "## 📚 Reference Details\n\n"
    for v in verified:
        ref = v["reference"]
        craap = v["craap"]
        name = ref.get("platform_name") or ref.get("url", "unknown")[:40]
        score = craap.get("overall_score", "?")
        summary = craap.get("summary", "No summary")
        
        # Score emoji
        if score and isinstance(score, (int, float)):
            if score >= 4:
                score_emoji = "✅"
            elif score >= 2.5:
                score_emoji = "⚠️"
            else:
                score_emoji = "❌"
        else:
            score_emoji = "❓"
        
        details += f"### {score_emoji} {name} — {score}/5\n"
        details += f"*Type: {ref.get('source_type', 'unknown')}*"
        if ref.get("stated_author"):
            details += f" | *Author: {ref['stated_author']}*"
        if ref.get("url"):
            details += f" | [Link]({ref['url']})"
        details += "\n\n"
        details += f"**Summary:** {summary}\n\n"
        
        if craap.get("currency"):
            details += f"| Criterion | Score | Evidence |\n"
            details += f"|-----------|-------|----------|\n"
            details += f"| **C**urrency | {craap['currency']['score']}/5 | {craap['currency'].get('evidence', 'N/A')[:80]}... |\n"
            details += f"| **R**elevance | {craap['relevance']['score']}/5 | {craap['relevance'].get('evidence', 'N/A')[:80]}... |\n"
            details += f"| **A**uthority | {craap['authority']['score']}/5 | {craap['authority'].get('evidence', 'N/A')[:80]}... |\n"
            details += f"| **A**ccuracy | {craap['accuracy']['score']}/5 | {craap['accuracy'].get('evidence', 'N/A')[:80]}... |\n"
            details += f"| **P**urpose | {craap['purpose']['score']}/5 | {craap['purpose'].get('evidence', 'N/A')[:80]}... |\n\n"
        
        if craap.get("red_flags"):
            details += f"🚩 {', '.join(craap['red_flags'])}\n\n"
        
        details += "---\n\n"
    
    yield header, details

# ============== GRADIO APP ==============

def create_app(volume=None):
    """Create the Snifftest Gradio app."""
    
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
    )
    
    def run_with_volume(text, progress=gr.Progress()):
        """Wrapper to inject volume into pipeline."""
        for status, result in run_snifftest(text, volume=volume, progress=progress):
            yield status, result
    
    with gr.Blocks(theme=theme, title="👃 Snifftest") as demo:
        gr.Markdown("""
        # 👃 Snifftest
        ### Does this AI response pass the smell test?
        
        Paste any AI-generated text below. Snifftest will extract the references 
        and evaluate each one using the **CRAAP framework** (Currency, Relevance, 
        Authority, Accuracy, Purpose).
        
        ⚠️ *Note: Each reference takes ~5 seconds to verify. A response with 10 references may take ~1 minute.*
        
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
            fn=run_with_volume,
            inputs=[input_text],
            outputs=[status, result],
        )
        
        gr.Markdown("""
        ---
        **Snifftest Ratings:** 🌟 Sweet (4+) | 😊 Fresh (3-4) | 😬 Funky (2.5-3) | 🤢 Foul (<2.5)
        
        **CRAAP Framework:** **C**urrency · **R**elevance · **A**uthority · **A**ccuracy · **P**urpose
        """)
    
    return demo