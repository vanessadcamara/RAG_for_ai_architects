import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import process_documents
from src.embeddings import verify_ollama, generate_embeddings_batch
from src.indexing import get_colection, index_chunks, list_fonts
from src.retrieval import retrieve_chunks
from src.generation import answer, print_result
from config import DADOS_DIR



def run_indexing(logger): 
    ''' Executes pipeline: 
    1. read docs
    2. split in chunks
    3. generate embeddings
    4. saves in chromaDB
    '''

    if not verify_ollama():
        logger.error("Ollama is not running. Please start Ollama and try again.")
        sys.exit(1)        
    
    chunks = process_documents(DADOS_DIR, logger)
    logger.info(f"Generated {len(chunks)} chunks from documents.")

    chunks_with_embeddings = generate_embeddings_batch(chunks, logger)
    logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks.")

    collection = get_colection(logger)
    index_chunks(collection, chunks_with_embeddings, logger)
    logger.info("Indexing completed successfully.")

    for font in list_fonts(collection, logger):
        logger.info(f"Indexed font: {font}")
    
def run_retrieval(query, logger): 
    '''
    1. convert query to embedding
    2. retrieve similar chunks from chromaDB
    3. generate answer with Mistral AI
    '''

    if not verify_ollama():
        logger.error("Ollama is not running. Please start Ollama and try again.")
        sys.exit(1)

    collection = get_colection(logger)
    if collection.count() == 0:
        logger.error("No chunks indexed. Please run the indexing pipeline first.")
        sys.exit(1)
    
    fonts = list_fonts(collection, logger)

    if query: 
        chunks = retrieve_chunks(collection, query, logger)
        answer_query = answer(chunks, query, logger)
        print_result(answer_query, fonts, logger)
        return
    
    print("Digite sua pergunta (ou 'sair' para encerrar):\n")

    
    while True: 
        try: 
            question = input(">> ")
            if not question: 
                continue
            if question.lower() in ("sair", "exit", "quit", "q"):
                print("Até mais!\n")
                break

            # Pipeline RAG
            # converts question to embedding, retrieves similar chunks from chromaDB, generates answer with Mistral AI
            chunks = retrieve_chunks(collection, question, logger)
            resultado = answer(chunks, question, logger)
            print_result(resultado, fonts, logger)
            print()
 
        except KeyboardInterrupt:
            print("\n\n👋 Interrompido pelo usuário.\n")
            break
        except Exception as e:
            print(f"Erro: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG For AI Engineers (LLMs, MLOps e n8n)"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Executa o pipeline de indexação dos documentos"
    )
    parser.add_argument(
        "--ask",
        type=str,
        default=None,
        help="Faz uma única pergunta (sem modo interativo)"
    )
    args = parser.parse_args()
 
    if args.index:
        run_indexing()
    else:
        run_retrieval(query=args.ask)

if __name__ == "__main__":
    main()
 

