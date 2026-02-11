# Spiritual Guide RAG System

A complete Retrieval-Augmented Generation (RAG) system that reads your `spiritual.md` document and answers questions about Indian spiritual sites, temples, and pilgrimage destinations.

## What Is This?

This system combines:
- **Document Loading**: Reads your spiritual.md file
- **Embeddings**: Converts text into numerical representations for semantic search
- **Vector Store**: FAISS database for fast similarity search
- **LLM Integration**: Uses either Ollama (local/free) or OpenAI API
- **Question Answering**: RAG chain that finds relevant passages and generates answers

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
cd c:\Sandbox\endee\my_ai_project
pip install -r requirements.txt
```

### 2. Set Up an LLM

**Option A: Ollama (Free, Local)**
```bash
# Download: https://ollama.ai
# Start: ollama serve
# Pull model: ollama pull mistral
```

**Option B: OpenAI (Requires API Key)**
```bash
$env:OPENAI_API_KEY = "sk-your-key-here"
```

### 3. Run the System
```bash
python run_rag.py
```

### 4. Ask Questions
```
You: What is the Atma Linga?
Assistant: [Detailed answer about the Atma Linga from your document]
```

See [QUICK_START.md](QUICK_START.md) for more details.

## 📁 Project Structure

```
my_ai_project/
├── spiritual.md                 # Your source document 📄
├── 
├── CORE MODULES:
├── data_loader.py              # Load and chunk markdown
├── rag_system.py               # Main RAG implementation
├── 
├── USAGE:
├── run_rag.py                  # Interactive interface ⭐ START HERE
├── examples.py                 # Code examples
├── 
├── UTILITIES:
├── utils.py                    # Analysis, export tools
├── config.py                   # Configuration settings
├── 
├── DOCUMENTATION:
├── README.md                   # This file
├── QUICK_START.md              # 5-minute quick start
├── SETUP_GUIDE.md              # Detailed setup guide
├── .env.example                # Environment template
├── 
├── OUTPUTS (generated):
├── faiss_index/                # Vector store (created after first run)
├── rag_responses.json          # Saved Q&A pairs
├── chunks.txt                  # Debug chunks
└── requirements.txt            # Dependencies

```

## 📚 File Guide

| File | Purpose | Use When |
|------|---------|----------|
| `run_rag.py` | Interactive chat interface | You want to ask questions | 
| `examples.py` | Code examples & patterns | You want to see how to use the API |
| `data_loader.py` | Data loading & chunking | You need to understand text processing |
| `rag_system.py` | Main RAG system | You want to integrate into your code |
| `utils.py` | Export & analysis tools | You want to save answers to CSV/HTML |
| `config.py` | Configuration settings | You want to customize behavior |
| `QUICK_START.md` | 5-minute setup | You're new and want to start fast |
| `SETUP_GUIDE.md` | Detailed instructions | You need complete setup documentation |

## 💡 Key Features

✅ **Complete RAG Pipeline**
- Load any markdown file
- Automatic text chunking
- Semantic embeddings with HuggingFace
- FAISS vector search
- LLM integration (Ollama or OpenAI)

✅ **Multiple Interfaces**
- Interactive CLI chat
- Batch question answering
- Programmatic API
- Web-ready export formats

✅ **Flexible LLM Backends**
- Ollama (open-source, runs locally)
- OpenAI API (gpt-3.5-turbo, gpt-4)
- Easy to extend for other models

✅ **Advanced Features**
- Similarity search without LLM
- Custom prompt templates
- Response caching and export
- Analysis and metrics
- Multiple output formats (JSON, CSV, Markdown, HTML)

## 🔧 Common Tasks

### Ask a Single Question
```python
from rag_system import SpiritualRAG

rag = SpiritualRAG(llm_type="ollama")
rag.load_and_embed_data()

response = rag.answer_question(
    "What are the 12 Jyotirlingas?"
)
print(response['answer'])
```

### Batch Process Multiple Questions
```python
questions = [
    "What is the Atma Linga?",
    "Tell me about Shakti Peethas",
    "Which temple has a 57-foot statue?",
]

responses = rag.batch_questions(questions)
rag.save_responses(responses, "answers.json")
```

### Search Without LLM (Fast)
```python
results = rag.similarity_search("temples with statues", k=5)
for content, score in results:
    print(f"Score: {score:.4f}\nContent: {content}\n")
```

### Export Answers to Different Formats
```python
from utils import ResponseExporter

# Load previously saved responses
with open("rag_responses.json") as f:
    responses = json.load(f)

# Export to multiple formats
ResponseExporter.to_csv(responses)        # → responses.csv
ResponseExporter.to_markdown(responses)   # → responses.md
ResponseExporter.to_html(responses)       # → responses.html
```

### Run Interactive Chat
```bash
python run_rag.py
```

## 🎯 Example Questions

Try asking:
- "What is the Atma Linga and where is it?"
- "Tell me about the 12 Jyotirlingas of Lord Shiva"
- "What are Shakti Peethas?"
- "Where is Udupi Sri Krishna Mutt?"
- "Which temple has the world's second-tallest Shiva statue?"
- "What is unique about Dharmasthala Manjunatha Temple?"
- "Tell me about Kashi Vishwanath temple"
- "What Jyotirlinga is in Uttarakhand?"

## 🛠️ Customization

### Change Embedding Model
```python
rag = SpiritualRAG(
    embeddings_model="sentence-transformers/all-mpnet-base-v2"
)
```

### Change LLM Model
- **Ollama**: Install a new model (`ollama pull llama2`)
- **OpenAI**: Change model in `config.py`

### Adjust Chunk Size
In `rag_system.py`:
```python
rag.load_and_embed_data(chunk_size=1000, chunk_overlap=200)
```

### Custom System Prompt
Modify the template in `_create_qa_chain()` method in `rag_system.py`

## 📊 Performance

- **Embedding Creation**: 2-3 seconds per 100 chunks (depends on CPU)
- **Query Processing**: 1-2 seconds per question (Ollama), <1 second (OpenAI)
- **Vector Search**: <100ms
- **Memory Usage**: ~100-500MB depending on data size
- **Storage**: FAISS index ~10-50MB

## 🐛 Troubleshooting

### "Could not connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify model is installed
ollama list
ollama pull mistral
```

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "FAISS index loading fails"
```bash
python run_rag.py --reload
```

### Slow First Run
First run takes 1-2 minutes to create embeddings. Subsequent runs are instant!

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Start here! 5-minute setup
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation & configuration
- **[examples.py](examples.py)** - 9 code examples showing different use cases

## 🔌 Architecture

```
spiritual.md
     ↓
[DataLoader: Chunk text]
     ↓
[HuggingFace: Create embeddings]
     ↓
[FAISS: Build vector index]
     ↓
━━━━━━ (User asks question) ━━━━━━
                ↓
        [Vector search: Find similar chunks]
                ↓
        [Retrieve: Get top K chunks]
                ↓
        [LLM: Generate answer with context]
                ↓
            Answer!
```

## 💻 System Requirements

- Python 3.8+
- 2GB RAM minimum (4GB+ recommended)
- ~100MB disk space for embeddings
- CPU: Any processor (GPU optional but faster)
- Internet: Only for OpenAI API or first Huggingface model download

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Suggestions and improvements welcome!

## 🔗 Related Projects

- [LangChain](https://python.langchain.com) - RAG framework
- [Ollama](https://ollama.ai) - Local LLM
- [Sentence Transformers](https://www.sbert.net) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search

## 📞 Support

For help with:
- **Setup issues**: See SETUP_GUIDE.md troubleshooting section
- **Ollama**: https://ollama.ai
- **OpenAI**: https://platform.openai.com/docs
- **LangChain**: https://python.langchain.com

---

**Happy learning! 🙏**

Start with `python run_rag.py` and start asking questions!
