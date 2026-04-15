# RAG_for_ai_architects

## 📁 Estrutura do Projeto

```
RAG_for_ai_engineers/
├── 📂 dados/                    ← Arquivos .txt e .pdf
├── 📂 chroma_db/                ← Auto-gerado na indexação
├── 📂 src/
│   ├── 🔵 etapa_01__ingestion.py     ← Leitura + chunking
│   ├── 🟢 etapa_02__embeddings.py    ← Vetorização com Ollama
│   ├── 🟡 etapa_03__indexing.py      ← Persistência no ChromaDB
│   ├── 🟠 etapa_04__retrieval.py     ← Busca semântica
│   └── 🔴 etapa_05__generation.py    ← Prompt otimizado + Mistral
├── 📄 config.py                 ← Todos os parâmetros centralizados
├── 📄 pipeline.py               ← Orquestrador principal
└── 📄 requirements.txt           ← Dependências do projeto
```

### 📋 Descrição dos Componentes

| Arquivo | Descrição |
|---------|-----------|
| **dados/** | Diretório para armazenar arquivos .txt e .pdf a serem indexados |
| **chroma_db/** | Banco de dados vetorial (auto-gerado durante indexação) |
| **etapa_01__ingestion.py** | Leitura de documentos + chunking automático |
| **etapa_02__embeddings.py** | Vetorização de textos usando modelo Ollama |
| **etapa_03__indexing.py** | Persistência dos embeddings no ChromaDB |
| **etapa_04__retrieval.py** | Busca semântica de contextos relevantes |
| **etapa_05__generation.py** | Geração de respostas com prompt otimizado |
| **config.py** | Configurações centralizadas de parâmetros |
| **pipeline.py** | Orquestrador que controla o fluxo completo |

# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar os modelos (uma vez)
ollama pull nomic-embed-text
ollama pull mistral

# 3. Colocar seus arquivos em dados/ e indexar
python pipeline.py --indexar

# 4. Fazer perguntas
python pipeline.py
python pipeline.py -p "O que é MLflow?"