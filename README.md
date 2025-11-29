# ELIA - Explainable Language Model Interpretability Analysis

![ELIA Logo](LOGO/Logo.png)

ELIA is a comprehensive, interactive suite for analyzing and interpreting the internal mechanisms of Large Language Models (LLMs). It brings together three powerful interpretability techniques—Attribution Analysis, Function Vectors, and Circuit Tracing—into a single, user-friendly Streamlit interface.

Designed for researchers and developers, ELIA provides deep insights into how models like **OLMo-2-1124-7B** process information, make decisions, and evolve representations across layers.

## Key Features

### 1. Attribution Analysis & Influence Tracing
Understand *why* the model generated a specific token.
*   **Interactive Heatmaps**: Visualize token importance using Integrated Gradients, Occlusion, and Saliency methods (powered by `inseq`).
*   **Influence Tracing**: Automatically identify which documents in the training data (Dolma dataset) most influenced the generation using vector similarity search.
*   **AI-Powered Explanations**: Get natural language explanations of the attribution patterns using the Qwen API.
*   **Faithfulness Verification**: Automated checks to ensure the AI explanations accurately reflect the underlying data.

### 2. Function Vectors & Layer Evolution
Explore the "functional" geometry of the model's latent space.
*   **3D PCA Visualization**: Interact with a 3D plot of function vectors to see how different tasks (e.g., QA, summarization, classification) cluster in the model's internal representation.
*   **Layer Evolution**: Track how the internal representation of a prompt changes layer-by-layer, identifying where key transformations occur.
*   **Interactive Analysis**: Enter any text to see where it falls in the function vector space and how it evolves through the network.

### 3. Circuit Tracing
Dive into the specific sub-networks responsible for model behaviors.
*   **Interactive Circuit Graphs**: Visualize the computation graph of the model, showing connections between embeddings, attention heads, and MLP features.
*   **Subnetwork Discovery**: Isolate and analyze specific paths and components that contribute to a prediction.
*   **Feature Interpretation**: View human-readable interpretations of individual neurons and their activation patterns.

## Prerequisites

*   **Python 3.10+**
*   **MPS (Apple Silicon)** (recommended) or CUDA-capable GPU for acceleration.
*   **OLMo-2-1124-7B Model Weights**: You will need the weights for the OLMo model locally.
*   **Qwen API Key**: Required for generating natural language explanations and performing semantic verification. You need to obtain your own API key (see [Installation](#installation) step 5).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ELIA.git
    cd ELIA
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Weights:**
    Place the `OLMo-2-1124-7B` model files in the `models/` directory:
    ```
    models/
    └── OLMo-2-1124-7B/
        ├── config.json
        ├── pytorch_model.bin (or .safetensors)
        ├── tokenizer.json
        └── ...
    ```

4.  **Pre-computation & Setup:**
    
    ELIA requires several pre-computed assets to function fully. Please run the following steps in order:

    **a. Build Influence Tracer Index (Required for Attribution Analysis):**
    Creates a searchable FAISS index from the Dolma dataset sample for training data attribution.
    ```bash
    python3 influence_tracer/build_dolma_index.py
    ```

    **b. Generate Function Vectors (Required for Function Vectors Analysis):**
    Computes mean activation vectors for various task categories (English & German) to populate the 3D PCA visualization.
    ```bash
    python3 function_vectors/generate_function_vectors.py
    ```

    **c. Train Cross-Layer Transcoder (Required for Circuit Tracing):**
    Trains a JumpReLU Sparse Autoencoder (SAE) on the OLMo model to analyze feature circuits.
    ```bash
    python3 circuit_analysis/train_clt_and_plot.py
    ```
    *Note: This trains a small model on top of OLMo layers. It requires the Dolma dataset sample.*

5.  **API Configuration:**
    ELIA uses the Qwen API for generating natural language explanations. You must provide your own API key.
    
    Set the `QWEN_API_KEY` environment variable before running the app:
    ```bash
    export QWEN_API_KEY="your_qwen_api_key_here"
    ```
    *Alternatively, you can manually edit `utilities/utils.py` to insert your key, though using environment variables is recommended for security.*

## Usage

1.  **Start the Web Application:**
    You can use the provided launcher script:
    ```bash
    python3 run_webapp.py
    ```
    Or run via Streamlit directly:
    ```bash
    streamlit run web_app.py
    ```

2.  **Access the Interface:**
    Open your web browser and navigate to `http://localhost:8501`.

3.  **Explore:**
    Use the sidebar to switch between **Attribution Analysis**, **Function Vectors**, and **Circuit Tracing**.

## Project Structure

*   `web_app.py`: Main application entry point.
*   `attribution_analysis/`: Logic and UI for attribution heatmaps and influence tracing.
*   `function_vectors/`: PCA visualization and layer evolution analysis.
*   `circuit_analysis/`: Interactive circuit graphs and subnetwork exploration.
*   `influence_tracer/`: Scripts for indexing and searching the Dolma training dataset.
*   `utilities/`: Helper functions, localization, and survey components.
*   `locales/`: Translation files (currently supporting English).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
