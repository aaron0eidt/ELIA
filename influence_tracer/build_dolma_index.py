import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm as tqdm_iterator
import sys
import torch

# Configuration for the script.
DOLMA_DIR = os.path.join("influence_tracer", "dolma_dataset_sample_1.6v")
INDEX_DIR = os.path.join("influence_tracer", "influence_tracer_data")
INDEX_PATH = os.path.join(INDEX_DIR, "dolma_index_multi.faiss")
MAPPING_PATH = os.path.join(INDEX_DIR, "dolma_mapping_multi.json")
STATE_PATH = os.path.join(INDEX_DIR, "index_build_state_multi.json")
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# Performance tuning.
BATCH_SIZE = 131072
SAVE_INTERVAL = 10

def build_index():
    # Scans the Dolma dataset, creates vector embeddings, and builds a FAISS index.
    print("--- Starting Influence Tracer Index Build (Optimized for Speed) ---")

    if not os.path.exists(DOLMA_DIR):
        print(f"Error: Dolma directory not found at '{DOLMA_DIR}'")
        print("Please ensure the dolma_dataset_sample_1.6v directory is in your project root.")
        sys.exit(1)

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load or initialize the state to allow resuming.
    processed_files = []
    doc_id_counter = 0
    total_docs_processed = 0
    doc_mapping = {}

    if os.path.exists(STATE_PATH):
        print("Found existing state. Attempting to resume...")
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                state = json.load(f)
            processed_files = state.get('processed_files', [])
            doc_id_counter = state.get('doc_id_counter', 0)
            total_docs_processed = state.get('total_docs_processed', 0)

            with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
                doc_mapping = json.load(f)

            print(f"Reading existing index from {INDEX_PATH}...")
            index = faiss.read_index(INDEX_PATH)
            print(f"Resumed from state: {len(processed_files)} files processed, {total_docs_processed} documents indexed.")
        except (IOError, json.JSONDecodeError, RuntimeError) as e:
            print(f"Error resuming from state: {e}. Starting fresh.")
            processed_files = []
            doc_id_counter = 0
            total_docs_processed = 0
            doc_mapping = {}
            index = None  # Will be re-initialized
    else:
        print("No existing state found. Starting fresh.")
        index = None

    # Detect the best device to use (MPS, CUDA, or CPU).
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Load the sentence transformer model.
    print(f"Loading sentence transformer model: '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the required libraries are installed.")
        print("Try running: pip install sentence-transformers faiss-cpu numpy tqdm")
        sys.exit(1)
    print("Model loaded successfully.")

    # Initialize the FAISS index if it wasn't loaded.
    if index is None:
        embedding_dim = model.get_sentence_embedding_dimension()
        # Use Inner Product for cosine similarity.
        index = faiss.IndexFlatIP(embedding_dim)
        print(f"FAISS index initialized with dimension {embedding_dim} using Inner Product (IP) for similarity.")

    # Get a list of all files to process.
    print(f"Scanning for documents in '{DOLMA_DIR}'...")
    all_files = sorted([os.path.join(DOLMA_DIR, f) for f in os.listdir(DOLMA_DIR) if f.endswith('.json')])
    files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]

    if not files_to_process:
        if processed_files:
            print("✅ All files have been processed. Index is up to date.")
            print("--- Index Build Complete ---")
            return
        else:
            print(f"Error: No JSON files found in '{DOLMA_DIR}'.")
            sys.exit(1)
            
    print(f"Found {len(all_files)} total files, {len(files_to_process)} remaining to process.")

    # Process each file.
    print(f"Processing remaining files with batch size {BATCH_SIZE}...")

    files_processed_since_save = 0
    for file_idx, path in enumerate(tqdm_iterator(files_to_process, desc="Processing files")):
        texts_batch = []
        batch_doc_info = []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        if text:
                            texts_batch.append(text)
                            batch_doc_info.append({
                                'id': doc_id_counter,
                                'info': {
                                    'source': data.get('source', 'Unknown'),
                                    'file': os.path.basename(path),
                                    'text_snippet': text[:200] + '...'
                                }
                            })
                            doc_id_counter += 1

                            # Process the batch when it's full.
                            if len(texts_batch) >= BATCH_SIZE:
                                embeddings = model.encode(texts_batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                                index.add(embeddings.astype('float32'))

                                # Update the document mapping.
                                for doc in batch_doc_info:
                                    doc_mapping[str(doc['id'])] = doc['info']

                                total_docs_processed += len(texts_batch)
                                texts_batch = []
                                batch_doc_info = []
                    except json.JSONDecodeError:
                        continue

            # Process any remaining documents in the last batch.
            if texts_batch:
                embeddings = model.encode(texts_batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                index.add(embeddings.astype('float32'))

                # Update the mapping for the final batch.
                for doc in batch_doc_info:
                    doc_mapping[str(doc['id'])] = doc['info']

                total_docs_processed += len(texts_batch)

            # Save progress periodically.
            processed_files.append(os.path.basename(path))
            files_processed_since_save += 1

            if files_processed_since_save >= SAVE_INTERVAL or file_idx == len(files_to_process) - 1:
                print(f"\nSaving progress ({total_docs_processed} docs processed)...")
                faiss.write_index(index, INDEX_PATH)
                with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
                    json.dump(doc_mapping, f)

                current_state = {
                    'processed_files': processed_files,
                    'doc_id_counter': doc_id_counter,
                    'total_docs_processed': total_docs_processed
                }
                with open(STATE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(current_state, f)
                
                files_processed_since_save = 0
                print("Progress saved.")

        except (IOError) as e:
            print(f"Warning: Could not read or parse {path}. Skipping. Error: {e}")
            continue

    if index.ntotal == 0:
        print("Error: No text could be extracted from the documents. Cannot build index.")
        sys.exit(1)

    print(f"\n🎉 Total documents processed: {total_docs_processed}")
    print(f"✅ --- Index Build Complete ---")
    print(f"Created index for {index.ntotal} documents.")


if __name__ == "__main__":
    # This allows the script to be run from the command line.
    print("This script will build a searchable index from your Dolma dataset.")
    print("It needs to download a model and process all documents, so it may take some time.")
    
    # Check for required libraries.
    try:
        import sentence_transformers
        import faiss
        import numpy
        import tqdm
    except ImportError:
        print("\n--- Missing Required Libraries ---")
        print("To run this script, please install the necessary packages by running:")
        print("pip install sentence-transformers faiss-cpu numpy tqdm")
        print("---------------------------------\n")
        sys.exit(1)
        
    build_index() 