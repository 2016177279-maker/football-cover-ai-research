# Offline Streamlit proof of concept

This is a research PoC, not a production CTR predictor. It does not guarantee
views, engagement, ranking, or uplift.

## Modes

- `OFFLINE_DEMO` is the default. It reads only the synthetic feature fixture in
  `examples/` and builds deterministic evidence-guided prompts. It requires no
  YouTube API key, private dataset, thumbnail cache, checkpoint, or generator.
- `OPTIONAL_EXTERNAL_MODEL` exposes the same prompt for use with an external
  image provider. No provider integration or credential is bundled. Use only a
  service and content you are authorized to use.

Run from the repository root:

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

If an image-generation model is unavailable, the prompt and synthetic diagnostic
path remain usable. Generated candidates must preserve content semantics and be
reviewed by a human for factual accuracy, text readability, and visual artifacts.
Human decisions are not automatically fed into model training.
