# Web_Data_Project_DIA6_Royal_Family
This is the final project for the Web Data Mining and Semantics course at ESILV (DIA6). The project focuses on the 19th-century British Royal Family. It includes a full pipeline covering data acquisition, knowledge graph construction, semantic alignment, reasoning, and a Retrieval-Augmented Generation (RAG) system.

Team Members: Yuhan XUE, Anthony YANG, and François ZAPLETAL.

Repository Structure :
The project follows the mandatory organization to ensure clarity and reproducibility:
src/: Contains the five core Python modules (m1 to m5).
notebooks/: Contains the main execution pipeline (main_pipeline.ipynb).
data/: Stores raw extractions, processed CSV files, and KGE datasets (train, valid, test).
kg_artifacts/: Contains all RDF graphs, ontology files, and alignment scripts.
reports/: Contains the final report PDF and the video presentation (and the project visualizations in image/).
requirements.txt: Lists all necessary Python libraries for environment setup.
.gitignore: Ensures unnecessary local files are not tracked.

Installation :
To reproduce the environment, follow these steps:
Install the required libraries:pip install -r requirements.txt 
Download the SpaCy language model:python -m spacy download en_core_web_md
Setup Ollama for the RAG system:Download Ollama from the official website.Pull the model using the command: ollama run llama3:latest

How to Run : 
The project is designed to be entirely reproducible at any stage.

1. The Notebook Pipeline : 
The main entry point is notebooks/main_pipeline.ipynb.
Run in order: It is recommended to run the cells sequentially from Module 1 to Module 5 to see the full flow.
Isolated testing: You are not required to run everything at once. Because the repository already includes all intermediate data and knowledge graph files (CSV, TTL, OWL), you can test individual modules or cells separately.
Data updates: Note that running a specific cell will overwrite the existing data files in the repository with new results.

2. RAG System :
The final module allows for natural language questions over the knowledge graph. Ensure that the Ollama service is running in the background before executing these cells.
