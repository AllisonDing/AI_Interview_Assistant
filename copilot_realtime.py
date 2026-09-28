"""
AI Interview Copilot — Always Listening
-----------------------------------------
Upload resume → interviewer speaks → answer auto-generates on silence.
Behavioral/experience questions answered from resume (first person).
Technical questions answered by selected LLM.

Install:
    pip install nvidia-riva-client openai gradio>=4.0 numpy pypdf python-docx

Run:
    export NVIDIA_API_KEY=nvapi-...
    python copilot_realtime.py
"""

import os, io, wave, threading, queue
import numpy as np
import gradio as gr
import riva.client
from riva.client.auth import Auth
from openai import OpenAI

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import docx as python_docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── Config ───────────────────────────────────────────────────────────────────
NVIDIA_API_KEY      = os.environ.get("NVIDIA_API_KEY", "nvapi-YOUR_KEY_HERE")
WHISPER_SERVER      = "grpc.nvcf.nvidia.com:443"
WHISPER_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
LLM_BASE_URL        = "https://integrate.api.nvidia.com/v1"

MODELS = {
    "⚡ Nemotron Lightning 30B (fastest)": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "🧠 DeepSeek V4 Flash (smart + fast)": "deepseek-ai/deepseek-v4-flash-0731",
}
DEFAULT_MODEL_LABEL = "⚡ Nemotron Lightning 30B (fastest)"

DEEPSEEK_MODELS = {"deepseek-ai/deepseek-v4-flash-0731", "deepseek-ai/deepseek-v4-pro-0813"}

ASR_EVERY_N_CHUNKS = 4
STABLE_CYCLES      = 2
MIN_WORDS          = 5

FILLERS = {"um", "uh", "so", "yeah", "hmm", "okay", "ok", "like", "right", "well"}
PENDING_MARKER = "⏳ Generating answer..."

# ── Clients ───────────────────────────────────────────────────────────────────
whisper_auth = Auth(
    uri=WHISPER_SERVER, use_ssl=True,
    metadata_args=[
        ("function-id", WHISPER_FUNCTION_ID),
        ("authorization", f"Bearer {NVIDIA_API_KEY}"),
    ],
)
asr = riva.client.ASRService(whisper_auth)
llm = OpenAI(base_url=LLM_BASE_URL, api_key=NVIDIA_API_KEY)

answer_queue: queue.Queue = queue.Queue()

# ── Document reading (called once at upload) ──────────────────────────────────
def read_doc(path: str) -> str:
    if not path:
        return ""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if not HAS_PYPDF:
            return "[install pypdf: pip install pypdf]"
        reader = pypdf.PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext == ".docx":
        if not HAS_DOCX:
            return "[install python-docx: pip install python-docx]"
        doc = python_docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

# ── Audio helpers ─────────────────────────────────────────────────────────────
def to_wav(sr, data):
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.int16:
        data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(data.tobytes())
    return buf.getvalue()

def run_asr(chunks) -> str:
    if not chunks:
        return ""
    sr = chunks[0][0]
    combined = np.concatenate([d for _, d in chunks])
    cfg = riva.client.RecognitionConfig(
        language_code="en-US", max_alternatives=1,
        enable_automatic_punctuation=True, audio_channel_count=1,
    )
    resp = asr.offline_recognize(to_wav(sr, combined), cfg)
    return " ".join(
        a.transcript for r in resp.results for a in r.alternatives
    ).strip()

def is_filler(text: str) -> bool:
    words = text.lower().split()
    if len(words) < MIN_WORDS:
        return True
    return len([w for w in words if w not in FILLERS]) < 3

def placeholder_for(question: str) -> str:
    return f"Q: {question}\n\nA: {PENDING_MARKER}"

# ── Background LLM thread (receives resume TEXT directly) ─────────────────────
def _llm_thread(question: str, resume_text: str, model_id: str):
    try:
        print(f"[LLM] model={model_id} | resume_chars={len(resume_text)} | q={repr(question[:60])}")

        system_msg = """You are an AI interview coach helping a job candidate ace their interview in real time.

RULES:
1. For introduction, behavioral, experience, background, or personal questions
   (e.g. "Tell me about yourself", "Why do you want this role?", "Describe a challenge"):
   → Answer IN FIRST PERSON as the candidate, drawing from their resume.
   → Sound natural and confident. Keep it 2-4 sentences unless more detail helps.

2. For technical questions (algorithms, system design, coding, tools, frameworks):
   → Answer as a knowledgeable technical expert.
   → If the resume shows relevant experience, briefly connect it.

3. Never say you are an AI or reading from a resume. Speak as the candidate.
4. If the resume is missing, give a strong generic answer.
"""
        if resume_text.strip():
            system_msg += f"\n--- CANDIDATE RESUME ---\n{resume_text[:15_000]}"

        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": question},
            ],
            temperature=0.7, max_tokens=512, stream=True,
        )
        if model_id in DEEPSEEK_MODELS:
            kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}

        stream = llm.chat.completions.create(**kwargs)
        answer = "".join(
            c.choices[0].delta.content
            for c in stream
            if c.choices and c.choices[0].delta.content
        )
        if not answer.strip():
            answer = "[No response — model may be unavailable. Check API key.]"

        answer_queue.put(("answer", question, answer))

    except Exception as e:
        answer_queue.put(("error", question, str(e)))

# ── Stream handler ────────────────────────────────────────────────────────────
def on_chunk(chunk, state, resume_text, answer_current, model_label):
    if state is None:
        state = {
            "chunks": [], "tick": 0,
            "current_text": "", "prev_text": "",
            "stable_count": 0, "last_answered": "",
            "full_transcript": "",
        }

    if chunk is None:
        return state["full_transcript"], state, answer_current

    sr, data = chunk
    state["chunks"].append((sr, data))
    state["tick"] += 1

    new_answer = answer_current

    if state["tick"] % ASR_EVERY_N_CHUNKS == 0:
        try:
            new_text = run_asr(state["chunks"])
        except Exception:
            new_text = state["current_text"]
        state["current_text"] = new_text

        if new_text and new_text == state["prev_text"]:
            state["stable_count"] += 1
        else:
            state["stable_count"] = 0
        state["prev_text"] = new_text

        if (state["stable_count"] >= STABLE_CYCLES
                and new_text
                and new_text != state["last_answered"]
                and not is_filler(new_text)):

            state["last_answered"] = new_text
            state["full_transcript"] = (
                state["full_transcript"] + "\n" + new_text
            ).strip()
            state["chunks"] = []
            state["tick"] = 0
            state["stable_count"] = 0
            state["current_text"] = ""
            state["prev_text"] = ""

            ph = placeholder_for(new_text)
            new_answer = (answer_current + "\n\n---\n\n" + ph).strip() if answer_current.strip() else ph

            model_id = MODELS.get(model_label, MODELS[DEFAULT_MODEL_LABEL])
            print(f"[TRIGGER] '{new_text[:60]}' | resume_chars={len(resume_text or '')}")
            threading.Thread(
                target=_llm_thread,
                args=(new_text, resume_text or "", model_id),
                daemon=True,
            ).start()

    display = state["full_transcript"]
    live = state["current_text"]
    if live and live != state["last_answered"] and not is_filler(live):
        display = (display + "\n▶ " + live).strip()

    return display, state, new_answer

# ── Timer ─────────────────────────────────────────────────────────────────────
def poll_answers(current: str) -> str:
    updated = current
    changed = False
    while not answer_queue.empty():
        kind, question, body = answer_queue.get_nowait()
        ph = placeholder_for(question)
        real = f"Q: {question}\n\nA: {body}" if kind == "answer" else f"Q: {question}\n\nA: ⚠️ Error: {body}"
        updated = updated.replace(ph, real, 1) if ph in updated else (updated + "\n\n---\n\n" + real).strip()
        changed = True
    return updated if changed else current

# ── Upload: extract text immediately and store in state ───────────────────────
def on_upload(file):
    if file is None:
        return "", "No file loaded"
    try:
        text = read_doc(file)
        name = os.path.basename(file)
        print(f"[RESUME LOADED] {name}: {len(text)} chars")
        print(f"[RESUME PREVIEW] {text[:300]}")
        return text, f"✅  {name}  ({len(text):,} chars)"
    except Exception as e:
        return "", f"❌  {e}"

# ── Clear: keep resume, wipe transcript + answers only ───────────────────────
def clear_all(state):
    while not answer_queue.empty():
        answer_queue.get_nowait()
    blank = {
        "chunks": [], "tick": 0, "current_text": "", "prev_text": "",
        "stable_count": 0, "last_answered": "", "full_transcript": "",
    }
    return None, "", blank, ""

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
#transcript_box textarea {
    height: 220px !important;
    max-height: 220px !important;
    overflow-y: auto !important;
    resize: none;
}
#answer_box textarea {
    height: 400px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    resize: none;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AI Interview Copilot", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown(
        "# 🤖 AI Interview Copilot\n"
        "Upload your resume → interviewer asks a question → suggested answer appears automatically.\n\n"
        "**Behavioral / experience** → answered from your resume (first person)  \n"
        "**Technical questions** → answered by the selected LLM"
    )

    asr_state = gr.State({
        "chunks": [], "tick": 0, "current_text": "", "prev_text": "",
        "stable_count": 0, "last_answered": "", "full_transcript": "",
    })
    # Store extracted resume TEXT (not path) so it's always available
    resume_text_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📄 Upload Your Resume")
            upload_btn = gr.UploadButton(
                "📎  Upload Resume / Doc",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                variant="primary", size="lg",
            )
            doc_status = gr.Textbox(
                label="Loaded document", interactive=False,
                lines=1, placeholder="No document loaded yet",
            )

            gr.Markdown("### 🤖 LLM Model")
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL_LABEL,
                label="Answer model",
                interactive=True,
            )

            gr.Markdown("### 🎤 Leave mic on during the interview")
            mic = gr.Audio(
                sources=["microphone"], streaming=True, type="numpy",
                label="Listening — answer fires after ~4s of silence",
            )
            clear_btn = gr.Button("🗑️ Clear Transcript & Answers", variant="secondary")
            gr.Markdown(
                f"---\n"
                f"**Trigger:** {STABLE_CYCLES * ASR_EVERY_N_CHUNKS * 0.5:.0f}s silence  \n"
                f"**Min words:** {MIN_WORDS}"
            )

        with gr.Column(scale=2):
            transcript_box = gr.Textbox(
                label="🎙️ Full Conversation Transcript",
                lines=8, interactive=False,
                elem_id="transcript_box",
                placeholder="Everything said in the room appears here...",
            )
            answer_box = gr.Textbox(
                label="💡 Suggested Answers — read & speak as your own words",
                lines=16, interactive=False,
                elem_id="answer_box",
                placeholder="Suggested answers accumulate here as questions are detected...",
            )

    # Upload → extract text into state immediately
    upload_btn.upload(fn=on_upload, inputs=[upload_btn],
                      outputs=[resume_text_state, doc_status])

    # Stream passes resume TEXT (not path) directly to LLM thread
    mic.stream(fn=on_chunk,
               inputs=[mic, asr_state, resume_text_state, answer_box, model_dropdown],
               outputs=[transcript_box, asr_state, answer_box],
               stream_every=0.5)

    timer = gr.Timer(value=0.5)
    timer.tick(fn=poll_answers, inputs=[answer_box], outputs=[answer_box])

    # Clear only transcript + answers; resume stays intact
    clear_btn.click(fn=clear_all, inputs=[asr_state],
                    outputs=[mic, transcript_box, asr_state, answer_box])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)