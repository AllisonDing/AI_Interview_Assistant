# AI Interview Assistant

A real-time interview copilot. It listens to the conversation through your microphone, transcribes it, and suggests a short answer to each interviewer question, grounded in your resume.

- **Behavioral / experience questions** are answered in first person from your resume.
- **Technical questions** get a short, plain-spoken answer.
- **Coding requests** ("give me the code", "write it on the board") get code only, for the algorithm discussed earlier in the conversation.

## How it works

| Step | Component |
|---|---|
| Speech-to-text | Whisper Large v3, hosted on NVIDIA NVCF, called via `nvidia-riva-client` |
| Answer generation | `nvidia/nemotron-3-super-120b-a12b` via the NVIDIA API (OpenAI-compatible) |
| UI | Gradio, served locally at http://localhost:7860 |

1. The microphone streams audio in 0.5s chunks. Every 2s the buffered audio is sent to Whisper.
2. When the transcript stops changing for about 4s, the segment is treated as finished. It is added to the conversation transcript and sent to the model.
3. The model receives your resume, the last ~3,000 characters of the conversation, and the latest segment. If the segment is only you talking, it skips it. Otherwise it returns an answer.
4. The Suggested Answers box shows only the answer to the most recent question.

## Setup

Requires Python 3.10+ and an NVIDIA API key from [build.nvidia.com](https://build.nvidia.com).

```bash
pip install nvidia-riva-client openai "gradio>=4.0" numpy pypdf python-docx
export NVIDIA_API_KEY=nvapi-...
python copilot_realtime.py
```

The app exits with an error if `NVIDIA_API_KEY` is not set. Install the packages into the same Python you use to run the app. If `python-docx` or `pypdf` is missing, the resume upload shows ❌.

## Usage

1. Open http://localhost:7860.
2. Upload your resume (`.pdf`, `.docx`, `.txt`, or `.md`). Check that the character count looks right; a very small number means the text wasn't extracted.
3. Start recording on the microphone and leave it on.
4. **Clear Transcript & Answers** resets the conversation but keeps your resume.

Transcription and model errors are printed in the terminal as `[ASR ERROR]` and `[LLM RETRY]`.

## Configuration

Constants at the top of `copilot_realtime.py`:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Answer model |
| `ASR_EVERY_N_CHUNKS` | 4 | Transcribe every N × 0.5s chunks |
| `STABLE_CYCLES` | 2 | Unchanged transcriptions needed before answering (sets the ~4s pause) |
| `MIN_WORDS` | 5 | Shorter segments are ignored as filler |
| `MAX_SEGMENT_SEC` | 25 | Audio is split into pieces under Whisper's 30s window |
| `CONTEXT_CHARS` | 3000 | How much earlier conversation is sent with each question |

## Known limitations

- **Technical terms are often misheard.** Whisper transcribes words like GTC or SciPy incorrectly even from clean audio. The transcript box shows these errors; the model is instructed to interpret them using your resume, so answers are usually correct.
- **No speaker separation.** The microphone hears both you and the interviewer. The model decides from context which segments are questions, and it can occasionally skip a real question or answer your own speech.
- **Hosted API availability.** The NVIDIA API sometimes returns "Service temporarily overloaded". The app retries up to 3 times before showing an error.
- **Single user.** Answer state is kept in module-level variables, so run one browser session at a time.
