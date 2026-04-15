# RAG_for_ai_architects

## 📁 Estrutura do Projeto

```
RAG_for_ai_engineers/
├── 📂 dados/                    ← Arquivos .txt e .pdf
├── 📂 chroma_db/                ← Auto-gerado na indexação
├── 📂 src/
│   ├── etapa 01 ingestion.py     ← Leitura + chunking
│   ├── etapa 02 embeddings.py    ← Vetorização com Ollama
│   ├── etapa 03 indexing.py      ← Persistência no ChromaDB
│   ├── etapa 04 retrieval.py     ← Busca semântica
│   └── etapa 05 generation.py    ← Prompt otimizado + Mistral
├── 📄 config.py                 ← Todos os parâmetros centralizados
├── 📄 pipeline.py               ← Orquestrador principal
└── 📄 requirements.txt           ← Dependências do projeto
```

### 📋 Descrição dos Componentes

| Arquivo | Descrição |
|---------|-----------|
| **dados/** | Diretório para armazenar arquivos .txt e .pdf a serem indexados |
| **chroma_db/** | Banco de dados vetorial (auto-gerado durante indexação) |
| **ingestion.py** | Leitura de documentos + chunking automático |
| **embeddings.py** | Vetorização de textos usando modelo Ollama |
| **indexing.py** | Persistência dos embeddings no ChromaDB |
| **retrieval.py** | Busca semântica de contextos relevantes |
| **generation.py** | Geração de respostas com prompt otimizado |
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

# Q&A
### por que fazemos o chunk dos textos? 
-> porque o modelo tem um limite de tokens que consegue processar. Dessa forma, garantimos que todo 
texto é processado e recuperado em partes 

### por que fazemos a etapa de embeddings? 
-> para que seja possível realizar operações matemáticas entre vetores e recuperar o contexto mais 
adequado para a pergunta, utilizando similaridade entre esses vetores

### o que é o embedding? 
-> uma representação numérica do texto que pode ser recuperado no espaço vetorial

### Como funciona a busca vetorial (retrieval)
  1. A pergunta é convertida em vetor com o mesmo modelo de embedding
  2. O ChromaDB calcula a distância cosseno entre o vetor da pergunta
     e todos os vetores armazenados
  3. Os TOP_K chunks com menor distância (mais similares) são retornados
 