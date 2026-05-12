# summarize-session-slop

Point it at a recording of your D&D session. Get back a structured Markdown
summary of what actually happened in the story.

Entirely local. Whisper transcribes, a local Ollama model summarises, nothing
leaves the machine.

## Why local matters here

A four-hour session recording is four hours of your friends talking. Some of it
is the campaign. Most of it is arguing about rules, ordering food, and bits.
Uploading all of that to a hosted API to get a paragraph back is a poor trade —
it is slow, it costs money per session, and it means handing a recording of
private conversations to a third party.

Both halves of this run on hardware you already own.

## What it does

```bash
python dnd_scribe.py session1.mkv
```

1. **Transcribe** — OpenAI Whisper (`medium` by default) over the raw recording.
   Uses `fp16` automatically when CUDA is available and falls back to fp32 on
   CPU. `condition_on_previous_text=False` is set deliberately: Whisper's default
   is to feed prior text forward as context, which on long multi-speaker audio
   makes it lock into repetition loops and hallucinate whole passages. Turning it
   off costs a little coherence and buys a transcript that does not invent a
   paragraph and then say it four more times.

2. **Summarise** — the transcript goes to a local model via Ollama. If you do
   not pass `-m`, it queries your Ollama instance and gives you a menu of what
   you have installed.

The system prompt does the actual work. It instructs the model to **separate
in-character narrative from out-of-character table talk** and then emit fixed
sections:

- Session Overview
- Locations Visited
- NPCs Met / Interacted With
- Key Plot Points & Lore
- Loot & Items
- Combat Outcomes

with explicit instruction to discard scheduling, dice math, food, and jokes. The
fixed section list is the important part — it turns "summarise this" into a
consistent shape, so session 7's notes are comparable to session 3's instead of
being whatever the model felt like that day.

## Usage

```bash
pip install -r requirements.txt        # openai-whisper, ollama
python dnd_scribe.py session1.mkv      # interactive model picker
python dnd_scribe.py session1.mkv -m llama3.1:8b
```

Requires [Ollama](https://ollama.com) running locally with at least one model
pulled, and `ffmpeg` on PATH (Whisper needs it to decode media).

Whisper model size is the `WHISPER_MODEL_NAME` constant at the top of the file —
`tiny` / `base` / `small` / `medium` / `large-v3`. `medium` is the sweet spot for
multi-speaker room audio; `large-v3` is better and considerably slower.

## Example output

`session1_summary.md` and `session1_normalized_summary.md` are real output from a
real session, committed as examples of what comes out the other end.

The raw transcripts are **gitignored**, and so is any media. They are recordings
of actual people who did not sign up to be a sample dataset — the summaries show
what the tool does without shipping four hours of a friend doing a voice.

## Honest limitations

- **No speaker diarisation.** Whisper produces one undifferentiated wall of
  text, so the model infers who was talking from context. It is usually right
  about the narrative and frequently wrong about attribution. Adding
  `pyannote.audio` is the obvious upgrade and a real project in itself.
- **Context window.** A long session can exceed a small local model's context.
  There is no chunk-and-reduce pass — it sends the transcript in one shot, so
  with an 8k-context model a four-hour session will get truncated. Fixing this
  properly means map-reduce summarisation.
- Quality is bounded by the local model. A 7B will produce something serviceable
  and bland; the sections stay right, the prose does not sparkle.
- Room audio is hard. Crosstalk, dice clatter, and someone eating crisps near
  the mic all degrade the transcript, and a bad transcript summarises badly.
- No tests, no error recovery beyond "print the exception and stop."
