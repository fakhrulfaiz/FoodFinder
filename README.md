# 🍽️ FoodFinder - AI Restaurant Recommendation Agent

A multimodal ReAct agent for intelligent restaurant recommendations using RAG (Retrieval-Augmented Generation) with text and image search capabilities.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53+-red.svg)

## ✨ Features

- 🤖 **ReAct Agent** - Reasoning and Acting pattern for intelligent decision-making
- 🔍 **Text Search** - Find restaurants by cuisine, location, ratings, and more
- 🖼️ **Image Search** - Upload food/restaurant images to find visually similar places
- 💬 **Image Q&A** - Ask questions about food images using CLIP
- 📊 **LangSmith Integration** - Full observability and tracing
- 🎨 **Streamlit UI** - Clean, ChatGPT-like interface

## 🎨 Streamlit UI
<img width="890" height="913" alt="image" src="https://github.com/user-attachments/assets/86d1acea-ac4e-4ef9-809c-12da089da4b6" />


## 🏗️ Architecture

```
User Input → Streamlit UI → ReAct Agent → Tools → Vector Store
                                ↓           ↓
                          LangSmith    FAISS Index
                                         (Text + Images)
```

### ReAct Loop

The agent follows a reasoning-acting cycle:

1. **Reason** - Analyze what information is needed
2. **Act** - Call appropriate tools (text search, image search, image QA)
3. **Observe** - Review results
4. **Repeat** - Until sufficient information is gathered

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/fakhrulfaiz/FoodFinder.git
   cd FoodFinder
   ```
2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

   Required variables:

   ```
   OPENAI_API_KEY=your_openai_key
   LANGSMITH_API_KEY=your_langsmith_key  # Optional
   ```
5. **Prepare data and indexes**

   Place your Yelp dataset in `data/raw/` and run:

   ```bash
   python scripts/process_data.py
   ```

   This script will:
   - Process Yelp business and photo data
   - Generate embeddings for text (restaurant metadata) and images
   - Build FAISS indices for fast similarity search
   - Save processed data to `data/processed/`
   - Create index files in `indexes/` directory

   **Expected output:**
   ```
   indexes/
   ├── text_index.faiss          # Text embeddings index
   ├── text_metadata.pkl          # Restaurant metadata
   ├── image_index.faiss          # Image embeddings index
   └── image_metadata.pkl         # Image metadata
   ```

   **Note:** This process may take several minutes depending on dataset size.

### Running the App

**Streamlit Chat Interface:**

```bash
streamlit run frontend/chat_app.py
```

Open your browser to `http://localhost:8501`

## Usage

### Text Queries

```
"Find Italian restaurants in Philadelphia"
"What are the best pizza places?"
"Show me highly rated sushi restaurants"
```

### Image Queries

1. Upload a food/restaurant image
2. Ask questions:
   - "What type of cuisine is this?"
   - "Find similar restaurants"
   - "Describe this dish"

### Combined Queries

The agent automatically combines tools when needed:

- Upload pizza image → Agent finds similar restaurants → Agent fetches full details

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini
- **Framework**: LangChain + LangGraph
- **Embeddings**:
  - Text: `sentence-transformers/all-mpnet-base-v2`
  - Images: `openai/clip-vit-base-patch32`
- **Vector Store**: FAISS
- **Frontend**: Streamlit
- **Observability**: LangSmith

## Project Structure

```
FoodFinder/
├── frontend/
│   ├── chat_app.py          # Streamlit chat interface
│   └── README.md
├── src/
│   ├── agent/
│   │   ├── main_agent.py    # ReAct agent
│   │   └── tools/           # Agent tools
│   ├── embeddings/          # Text & image embedders
│   ├── vectorstore/         # FAISS index & retriever
│   └── utils/
├── scripts/
│   └── process_data.py      # Data processing
├── data/                    # Dataset (gitignored)
├── indexes/                 # FAISS indexes (gitignored)
├── tests/
└── requirements.txt
```

## Tools

The agent has access to three specialized tools:

1. **rag_text_search** - Search restaurants by text query
2. **rag_image_search** - Find visually similar restaurants
3. **image_qa** - Answer questions about images using CLIP

## Development

### Adding New Tools

1. Create tool in `src/agent/tools/`
2. Add to `CustomToolkit` in `toolkit.py`
3. Update agent system prompt

## Acknowledgments

- Yelp Dataset
- LangChain team
- OpenAI for GPT and CLIP models
- Streamlit team

---

**Built with ❤️ using LangChain and Streamlit**
