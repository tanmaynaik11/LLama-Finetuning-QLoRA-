# 🧠 Fine-Tuning & Evaluation of LLaMA 3.2 1B with QLoRA, LangFlow, and DeepEval

This project fine-tunes Meta’s LLaMA 3.2 1B using **QLoRA** and evaluates instruction dataset quality using **LLM-as-a-judge**, integrated with **LangFlow**, **Docling**, and **DeepEval**. It streamlines dataset scoring, model tuning, and response evaluation in a lightweight and efficient workflow suitable for consumer GPUs.

---

## 🚀 Project Highlights

- ✅ Fine-tuned **LLaMA 3.2 1B** using **4-bit QLoRA + LoRA adapters** on a single 24 GB GPU
- ✅ Built an LLM-powered auto-evaluator pipeline using **LiteLLM + Pydantic schemas**
- ✅ Filtered **5,000+ instruction-response pairs**, retaining **~91% high-quality data**
- ✅ Integrated **LangFlow** and **Docling** for visual scoring, auditing, and real-time analysis
- ✅ Evaluated fine-tuned model using **DeepEval** across fluency, coherence, and factuality

---

## 🛠️ Tech Stack

| Tool        | Purpose                          |
|-------------|----------------------------------|
| 🤖 LLaMA 3.2 1B | Base model for fine-tuning     |
| 🧠 QLoRA + LoRA | Efficient adapter fine-tuning  |
| ⚙️ Transformers | Tokenizer & model management    |
| 🔗 LiteLLM     | LLM-as-a-judge evaluation API  |
| 🌐 LangFlow    | Flow-based interface for prompt modeling & scoring |
| 📊 Docling     | Visual analysis of record quality |
| 📏 DeepEval    | Structured LLM evaluation metrics |
| 🐍 Python      | Core scripting and dataset curation |
| 💾 Datasets    | Custom instruction dataset      |

---

## 📈 Evaluation Metrics

| Metric               | Value                    |
|----------------------|--------------------------|
| 🎯 Match Accuracy     | 92.4% (DeepEval)         |
| 💬 Coherence Score    | 3.9 / 5 (DeepEval)       |
| 🧪 Instruction Retention | ~3,800 of 5,000+ pairs |
| ⚡ Inference Speed    | 2.3× faster vs. FP16     |
| 📚 Fluency Rating     | 4.7 / 5 peer-reviewed    |


