"""
AI Interview Copilot — Always Listening
-----------------------------------------
Upload resume → interviewer speaks → answer auto-generates on silence.
Behavioral/experience questions answered from resume (first person).
Technical questions answered by Nemotron Super.

Install:
    pip install nvidia-riva-client openai gradio>=4.0 numpy pypdf python-docx

Run:
    export NVIDIA_API_KEY=nvapi-...
    python copilot_realtime.py
"""

import os, io, wave, threading, queue, time
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
NVIDIA_API_KEY      = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise SystemExit("NVIDIA_API_KEY is not set. Run: export NVIDIA_API_KEY=nvapi-...")
WHISPER_SERVER      = "grpc.nvcf.nvidia.com:443"
WHISPER_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
LLM_BASE_URL        = "https://integrate.api.nvidia.com/v1"

LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"

ASR_EVERY_N_CHUNKS = 4
MAX_SEGMENT_SEC    = 25
CONTEXT_CHARS      = 3000
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
llm = OpenAI(base_url=LLM_BASE_URL, api_key=NVIDIA_API_KEY, timeout=20, max_retries=0)

answer_queue: queue.Queue = queue.Queue()
latest_question = ""
last_real_answer = ""

# ── Document reading (called once at upload) ──────────────────────────────────
def read_doc(path: str) -> str:
    if not path:
        return ""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if not HAS_PYPDF:
            raise RuntimeError("pypdf not installed: pip install pypdf")
        reader = pypdf.PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext == ".docx":
        if not HAS_DOCX:
            raise RuntimeError("python-docx not installed: pip install python-docx")
        doc = python_docx.Document(path)
        # Resume templates often put content in tables, which doc.paragraphs skips
        lines = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    lines.append(cell.text)
        return "\n".join(l for l in lines if l.strip())
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
        language_code="en", max_alternatives=1,
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

# ── Background LLM thread (receives resume TEXT directly) ─────────────────────
def _llm_thread(question: str, resume_text: str, context: str):
    try:
        print(f"[LLM] model={LLM_MODEL} | resume_chars={len(resume_text)} | q={repr(question[:60])}")

        system_msg = """You are an AI interview coach helping a job candidate ace their interview in real time.

OUTPUT FORMAT (strict):
- Give ONLY the answer the candidate should say out loud. No reasoning, no steps,
  no headings, no bullet lists, no bold, no preamble like "Here's an answer".
- 2-4 short, plain sentences. Conversational, confident, first person.
- Exception: when code is requested, follow rule 3 instead.

RULES:
1. Introduction, behavioral, experience, or background questions:
   answer in first person as the candidate, drawing from their resume.
2. Technical concept questions: give the direct answer in plain spoken language,
   and briefly connect it to the resume if relevant.
3. If the interviewer asks for code (e.g. "write it", "give me the code", "implement it",
   "LeetCode style", "on the board"): output ONLY a complete, correct Python solution
   in a ```python block, with no explanation before or after. Implement exactly the
   algorithm being discussed in the earlier conversation (e.g. "from scratch" means
   no scikit-learn; numpy is fine). Keep it clean and concise, like a whiteboard answer.
4. Never say you are an AI or reading from a resume.
5. If the resume is missing, give a strong generic answer; do not invent employer names.

INPUT NOTES:
- The input is a live speech-to-text transcript of the room. It often mishears names
  of tools, companies, and events (e.g. "Revit" for RAPIDS, "GDC" for GTC, "CPI" for SciPy).
  Silently interpret the intended words using the resume. Never mention or correct
  transcription errors.
- The transcript may include the candidate's own speech. Answer only the interviewer's
  latest question. If there is no interviewer question (it is just the candidate talking
  or small talk), reply with exactly: SKIP
"""
        if resume_text.strip():
            system_msg += f"\n--- CANDIDATE RESUME ---\n{resume_text[:15_000]}"

        kwargs = dict(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": (
                    f"Earlier conversation (context only, may be empty):\n\"\"\"{context}\"\"\"\n\n"
                    f"Latest transcript segment:\n\"\"\"{question}\"\"\"\n\n"
                    "Use the earlier conversation to understand what the latest segment refers to "
                    "(e.g. 'give me the code' refers to the algorithm discussed before).\n"
                    "Does the latest segment contain a question or request from the INTERVIEWER "
                    "(e.g. 'tell me about...', 'what is...', 'can you...')? "
                    "If the latest segment is only the candidate speaking (introducing themselves, "
                    "answering, 'I'm ...', 'I worked at ...'), reply with exactly SKIP. "
                    "Otherwise, reply with only the candidate's answer to the interviewer's latest question."
                )},
            ],
            temperature=0.3, max_tokens=800, stream=True,
        )
        last_err = None
        for attempt in range(3):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            try:
                stream = llm.chat.completions.create(**kwargs)
                answer = "".join(
                    c.choices[0].delta.content
                    for c in stream
                    if c.choices and c.choices[0].delta.content
                )
                if answer.strip():
                    answer_queue.put(("answer", question, answer))
                    return
                last_err = "empty response"
            except Exception as e:
                last_err = e
            print(f"[LLM RETRY] attempt {attempt + 1} failed: {str(last_err)[:120]}")
            time.sleep(1)

        answer_queue.put(("error", question, f"Model failed after 3 tries: {last_err}"))

    except Exception as e:
        answer_queue.put(("error", question, str(e)))

# ── Stream handler ────────────────────────────────────────────────────────────
def on_chunk(chunk, state, resume_text, answer_current):
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
        except Exception as e:
            print(f"[ASR ERROR] {e}")
            new_text = state.get("seg_text", "")
        state["seg_text"] = new_text
        # Whisper handles ~30s windows; bank long speech and start a fresh segment
        seg_sec = sum(len(d) for _, d in state["chunks"]) / state["chunks"][0][0]
        if seg_sec > MAX_SEGMENT_SEC and new_text:
            state["carry"] = (state.get("carry", "") + " " + new_text).strip()
            state["chunks"] = []
            state["seg_text"] = new_text = ""

        if state.get("carry"):
            new_text = (state["carry"] + " " + new_text).strip()
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
            context = state["full_transcript"][-CONTEXT_CHARS:]
            state["full_transcript"] = (
                state["full_transcript"] + "\n" + new_text
            ).strip()
            state["chunks"] = []
            state["carry"] = state["seg_text"] = ""
            state["tick"] = 0
            state["stable_count"] = 0
            state["current_text"] = ""
            state["prev_text"] = ""

            global latest_question
            latest_question = new_text
            new_answer = PENDING_MARKER

            print(f"[TRIGGER] '{new_text[:60]}' | resume_chars={len(resume_text or '')}")
            threading.Thread(
                target=_llm_thread,
                args=(new_text, resume_text or "", context),
                daemon=True,
            ).start()

    display = state["full_transcript"]
    live = state["current_text"]
    if live and live != state["last_answered"] and not is_filler(live):
        display = (display + "\n▶ " + live).strip()

    return display, state, new_answer

# ── Timer ─────────────────────────────────────────────────────────────────────
def poll_answers(current: str) -> str:
    global last_real_answer
    updated = current
    while not answer_queue.empty():
        kind, question, body = answer_queue.get_nowait()
        # Only show the answer to the most recent question; stale ones are dropped
        if question != latest_question:
            continue
        if kind == "answer" and body.strip().strip(".").upper() == "SKIP":
            updated = last_real_answer
        elif kind == "answer":
            updated = last_real_answer = body
        else:
            updated = f"⚠️ Error: {body}"
    return updated

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
    global latest_question, last_real_answer
    latest_question = last_real_answer = ""
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
        "**Technical questions** → answered by Nemotron Super"
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
                placeholder="The answer to the latest question appears here...",
            )

    # Upload → extract text into state immediately
    upload_btn.upload(fn=on_upload, inputs=[upload_btn],
                      outputs=[resume_text_state, doc_status])

    # Stream passes resume TEXT (not path) directly to LLM thread
    mic.stream(fn=on_chunk,
               inputs=[mic, asr_state, resume_text_state, answer_box],
               outputs=[transcript_box, asr_state, answer_box],
               stream_every=0.5)

    timer = gr.Timer(value=0.5)
    timer.tick(fn=poll_answers, inputs=[answer_box], outputs=[answer_box])

    # Clear only transcript + answers; resume stays intact
    clear_btn.click(fn=clear_all, inputs=[asr_state],
                    outputs=[mic, transcript_box, asr_state, answer_box])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)