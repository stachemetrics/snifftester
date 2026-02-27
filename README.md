# SniffTester

A prototype tool for verifying the accuracy of AI-generated reports by systematically evaluating their sources and claims.

## How it works

SniffTester applies the [CRAAP framework](https://en.wikipedia.org/wiki/CRAAP_test) — an academic assessment method examining five dimensions:

- **Currency** — timeliness of the source
- **Relevance** — applicability to the claims being made
- **Authority** — credibility of the source
- **Accuracy** — verifiability of the claims
- **Purpose** — intent behind the publication

The process runs in two sequential steps: extraction of all references and citations from the document, followed by evaluation against these criteria using live web search. Each source receives a CRAAP score (1–5) and an overall **Snifftest** rating:

| Rating | Score | Meaning |
|--------|-------|---------|
| 🟢 Sweet | 3.5+ | High quality sources |
| 🟡 Fresh | 2.5–3.5 | Generally good |
| 🟠 Funky | 2.0–2.5 | Some weak sources |
| 🔴 Foul | <2.0 | Poor quality sources |

## Technical implementation

Built on Google's Gemini with search integration. The interface uses [Gradio](https://gradio.app) and is hosted on [Modal](https://modal.com).

## Limitations

- **Performance**: Citations are evaluated sequentially rather than in parallel, creating delays for longer documents
- **Context sensitivity**: The tool can struggle to assess whether a source applies to a specific claim vs. lending general authority
- **User experience**: The current submit-and-receive model lacks the inline feedback users expect from editing tools

## Planned improvements

- Parallel evaluation to reduce latency
- Real-time feedback as you write
- Inline highlighting of problematic content within the source document, similar to grammar-checking tools

## Running locally

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Then run:

```bash
python app/app.py
```

Gradio will serve the app at `http://localhost:7860`.

## Deploying to Modal

Install Modal and authenticate:

```bash
pip install modal
modal setup
```

Create a secret named `gemini-secret` with your API key:

```bash
modal secret create gemini-secret GEMINI_API_KEY=your_api_key_here
```

Deploy from the `app/` directory:

```bash
modal deploy app/deploy.py
```

Modal will provision the container, create the `snifftest-logs` volume, and return a public URL. Logs are written to the persistent volume at `/root/logs/snifftest_logs.jsonl`.

## License

MIT — Copyright 2026 stachemetrics
