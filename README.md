# Company Policy RAG Agent

A policy-focused Retrieval-Augmented Generation (RAG) assistant built with LangChain, LangGraph, Hugging Face embeddings, and Gradio. It answers customer-policy questions from PDF documents and provides access to internal employee HR policies only after employee authentication.

## Features

* **PDF ingestion:** Loads separate customer-policy and employee-policy PDFs using `PyPDFLoader`.
* **Semantic retrieval:** Splits documents into chunks and indexes them in separate in-memory vector stores.
* **Grounded answers:** Retrieves up to four relevant passages for a question and instructs the agent to answer from those passages, including source page numbers.
* **Employee authentication:** Offers an authentication tool for internal HR-policy access. The agent's available tools and system prompt change after authentication.
* **Conversation memory:** Uses LangGraph's `MemorySaver` for the Gradio interface, with a separate thread ID for each browser session.
* **Model fallback:** Configures a primary Groq model and an NVIDIA fallback through LangChain middleware.
* **Conversation summarization:** Summarizes longer conversations after the configured token threshold.
* **Gradio chat UI:** Provides an interactive interface with example policy questions.
* **LangGraph Studio support:** Exposes `RAG_Agent` for use by LangGraph development tools.

## How It Works

1. **Load policy documents.** The application reads the customer and employee PDF files from environment variables.
2. **Split documents.** Each PDF's pages are split into chunks of 1,000 characters with 200 characters of overlap.
3. **Create embeddings.** The `sentence-transformers/all-mpnet-base-v2` model converts the chunks into embeddings.
4. **Index each policy collection.** Customer and employee chunks are stored separately in `InMemoryVectorStore` instances.
5. **Choose the permitted tools.** Before authentication, the agent can search customer policies or attempt employee authentication. After successful authentication, the middleware exposes only the employee-policy retrieval tool.
6. **Retrieve and answer.** The agent searches the appropriate vector store, receives relevant passages, and is instructed to answer from the retrieved content and cite page numbers.
7. **Serve the chat.** Gradio passes each conversation to the agent using a session-specific thread ID. A model fallback is configured if the primary model fails.

## Project Structure

```text
RAG-System/
├── RAG_Agent.py   # RAG pipeline, retrieval tools, authentication, agent, and Gradio UI
├── models.py      # Expected module exporting groq_model and nvidia_model
├── README.md
└── .gitignore
```

**Note:** The current repository branch must include a `models.py` module that defines `groq_model` and `nvidia_model`. The agent imports these objects but does not configure the provider models inside `RAG_Agent.py`.

## Requirements

Use Python 3.10 or newer, with compatible versions of the following packages:

* `langchain`
* `langgraph`
* `langchain-community`
* `langchain-text-splitters`
* `langchain-huggingface`
* `sentence-transformers`
* `pypdf`
* `gradio`
* The provider integration packages required by your `models.py` configuration

Install the packages in a virtual environment. Exact provider packages and versions depend on your `models.py` implementation. A pinned `requirements.txt` is recommended for reproducible setup.

## Configuration

Set the paths to both policy PDFs before running the application.

### Environment Variables

| Variable            | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
| `CUSTOMER_PDF_PATH` | Path to the PDF containing customer-facing policies   |
| `EMPLOYEE_PDF_PATH` | Path to the PDF containing internal employee policies |
| `EMPLOYEE_NAME`     | Employee name used by the demo authentication flow    |
| `EMPLOYEE_PASSCODE` | Passcode used by the demo authentication flow         |

The employee name and passcode have demo defaults in the code, but **do not use these defaults in a real deployment**. Configure secrets through environment variables or a secret manager, and never commit credentials to Git.

Example configuration:

```env
CUSTOMER_PDF_PATH=./data/customer_policies.pdf
EMPLOYEE_PDF_PATH=./data/employee_policies.pdf
EMPLOYEE_NAME=your_employee_name
EMPLOYEE_PASSCODE=replace_with_a_secure_passcode
```

Replace the example values with your actual configuration. Keep real credentials out of source control.

## Run Locally

1. Clone the repository and enter the project directory.
2. Create and activate a Python virtual environment.
3. Install the dependencies required by `RAG_Agent.py` and your `models.py` configuration.
4. Add the customer and employee policy PDFs and configure the environment variables.
5. Ensure `models.py` exports valid `groq_model` and `nvidia_model` chat-model instances.
6. Start the Gradio application:

```bash
python RAG_Agent.py
```

Gradio will print the local address for the chat interface in the terminal.

## Example Questions

* How do I cancel my plan?
* What is the refund policy for domains?
* How long are backups kept after cancellation?
* Tell me about employee salary and benefits.

The first three questions are customer-policy examples. The final question should require successful employee authentication before internal HR policies become available.

## Security and Deployment Considerations

* **Demo authentication:** The code compares a supplied name and passcode against configured values. For production, use a proper identity provider or secure authentication system, protect against brute-force attempts, and avoid passing reusable secrets through chat where possible.
* **In-memory vector stores:** Both policy collections are rebuilt at startup and are not persisted between runs.
* **Internal document protection:** Keep employee-policy PDFs and their embeddings isolated from public or customer-facing access.
* **RAG reliability:** Prompts ask the model to use retrieved passages, but prompts alone cannot guarantee that every answer is correct. Add automated evaluation and tests before production use.
* **Sensitive data:** Review provider data-handling settings and avoid logging credentials or confidential policy content.
* **Model availability:** The configured fallback can help with provider failures, but it does not guarantee uninterrupted service.

## Current Limitations and Future Improvements

* Add a persistent vector database instead of in-memory vector stores.
* Add a dependency file with pinned package versions.
* Add automated tests for retrieval quality, authentication success and failure, and cross-session isolation.
* Add RAG evaluation, such as answer relevance and faithfulness checks, with a controlled retry strategy.
* Improve authentication and session management for production.
* Add a dedicated orchestration-facing function or API, as suggested by the TODOs in the source code.

## Technologies

Python · LangChain · LangGraph · Hugging Face Sentence Transformers · In-memory vector search · PyPDF · Gradio · Groq / NVIDIA model providers

## Author

**Mohamed Fathy**
