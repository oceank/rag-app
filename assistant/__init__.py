"""
RAG assistant.

This module provides the most functionality for RAG usage:
- LLM embeddings (OpenAI, Azure OpenAi and Hugging Face models are supported);
- Vectorsores and retrievers (ChromaDB);
- Chains (RAG chain with using and returning source documents);
- LLMs (OpenAI, Azure OpenAi and Hugging Face models are supported);

Load and split documents:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

# Index documents using Hugging Face model
```python
from assistant import get_chromadb, get_embeddings_model

embeddings_model = get_embeddings_model("thenlper/gte-base", "hf")
vectorstore = get_chromadb(embeddings_model)
vectorstore.add_documents(splits)
```

# Create RAG chain using Hugging Face model
```python
from assistant import get_llm, get_rag_chain, get_retriever

retriever = get_retriever(vectorstore)
llm = get_llm("google/flan-t5-base", "hf")
chain = get_rag_chain(retriever, llm)
```

# Invoke RAG chain
```python
chain.invoke("What is Task Decomposition?")
# {'source_documents': [...], 'question': ..., 'answer': ...}
```

# Clear previous indices
```python
vectorstore.delete_collection()
```

# Index documents using OpenAI model
```python
embeddings_model = get_embeddings_model("text-embedding-ada-002", "openai")
vectorstore = get_chromadb(embeddings_model)
vectorstore.add_documents(splits)
```

# Create RAG chain using OpenAI model
```python
retriever = get_retriever(vectorstore)
llm = get_llm("gpt-4", "openai")
chain = get_rag_chain(retriever, llm)
```

# Invoke RAG chain
```python
chain.invoke("What is Task Decomposition?")
# {'source_documents': [...], 'question': ..., 'answer': ...}
```
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast, get_args

import dotenv
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_community.chat_models import ChatOllama

from .exceptions import HFImportError
from .settings import guess_model_type
from .types import (
    LLM,
    Device,
    ModelType,
    PredefinedRelevanceScoreFn,
    RelevanceScoreFn,
    RetrieverSearchType,
)

dotenv.load_dotenv(override=True)

if sys.platform == "linux":
    # For Linux, `sqlite3` can be outdated, so use the installed Python package.
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Import `chromadb` when `sqlite3` is patched.
import chromadb  # noqa: E402
import chromadb.api.types  # noqa: E402


def parse_model_name(name: str) -> Tuple[str, ModelType]:
    if ":" in name:
        model_type, name = name.split(":", 1)
        assert model_type in get_args(ModelType)
    else:
        model_type = guess_model_type()
    model_type = cast(ModelType, model_type)
    return name, model_type


def _get_torch_device(device: Optional[Device] = None) -> Any:
    try:
        import torch
    except ImportError as e:
        raise HFImportError() from e
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "gpu":
        return torch.device("cuda:0")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(
        f"Device should be one of 'cpu', 'gpu' or `None`, but value {device} "
        f"of type {type(device)} received."
    )


def get_hf_embeddings_model(
    name: str, device: Optional[Device] = None, trust_remote_code: bool = False
) -> HuggingFaceEmbeddings:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise HFImportError() from e
    torch_device = _get_torch_device(device)
    encode_kwargs = {"normalize_embeddings": True}
    model_kwargs = {"device": torch_device, "trust_remote_code": trust_remote_code}
    return HuggingFaceEmbeddings(
        model_name=name, encode_kwargs=encode_kwargs, model_kwargs=model_kwargs
    )

def get_embeddings_model(
    name: str,
    model_type: ModelType,
    *,
    device: Optional[Device] = None,
    trust_remote_code: bool = False,
) -> Embeddings:
    if model_type == "azure":
        return AzureOpenAIEmbeddings(azure_deployment=name)
    if model_type == "openai":
        return OpenAIEmbeddings(model=name)
    if model_type == "hf":
        return get_hf_embeddings_model(name, device, trust_remote_code)
    if model_type == "ollama":
        return OllamaEmbeddings(model=name)
    raise TypeError(f"Unknown model type '{model_type}'.")


def get_embeddings_model_config(embeddings_model: Embeddings) -> Tuple[str, ModelType]:
    if isinstance(embeddings_model, AzureOpenAIEmbeddings):
        assert embeddings_model.deployment is not None
        return embeddings_model.deployment, "azure"
    if isinstance(embeddings_model, OpenAIEmbeddings):
        return embeddings_model.model, "openai"
    if isinstance(embeddings_model, HuggingFaceEmbeddings):
        return embeddings_model.model_name, "hf"
    if isinstance(embeddings_model, OllamaEmbeddings):
        return embeddings_model.model, "ollama"
    raise TypeError(f"Unknown model type `{type(embeddings_model)}`.")


def get_hf_llm(
    name: str,
    device: Optional[Device] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[str] = None,
) -> HuggingFacePipeline:
    try:
        import torch  # noqa: F401
        from transformers import pipeline
    except ImportError as e:
        raise HFImportError() from e
    torch_device = _get_torch_device(device)
    pipe = pipeline(
        model=name,
        device=torch_device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        max_length=2048,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def get_llm(
    name: str,
    model_type: ModelType,
    *,
    device: Optional[Device] = None,
    trust_remote_code: bool = False,
    torch_dtype: Optional[str] = None,
) -> LLM:
    if model_type == "azure":
        return AzureChatOpenAI(azure_deployment=name, temperature=0.0)
    if model_type == "openai":
        return ChatOpenAI(model=name, temperature=0.0)
    if model_type == "hf":
        return get_hf_llm(name, device, trust_remote_code, torch_dtype)
    if model_type == "ollama":
        return ChatOllama(model=name, temperature=0.0)
    raise TypeError(f"Unknown model type '{model_type}'.")


def get_llm_config(llm: LLM) -> Tuple[str, ModelType]:
    if isinstance(llm, AzureChatOpenAI):
        assert llm.deployment_name is not None
        return llm.deployment_name, "azure"
    if isinstance(llm, ChatOpenAI):
        return llm.model_name, "openai"
    if isinstance(llm, HuggingFacePipeline):
        return llm.pipeline.model, "hf"
    if isinstance(llm, ChatOllama):
        return llm.model, "ollama"
    raise TypeError(f"Unknown model type `{type(llm)}`.")


def _assert_embeddings_model_ok_for_chromadb(
    embeddings_model: Embeddings, persist_directory: Path, collection_name: str
) -> None:
    """
    If model config is exists in the given collection of the given ChromaDB,
    check its compatibility with the given model.
    """
    if not persist_directory.exists():
        # Vectorstore doesn't exist yet.
        return
    model_name, model_type = get_embeddings_model_config(embeddings_model)
    client_settings = chromadb.Settings(
        is_persistent=True, persist_directory=str(persist_directory)
    )
    client = chromadb.Client(client_settings)
    try:
        collection = client.get_collection(collection_name, embedding_function=None)
    except ValueError:
        # Collection doesn't exist in the ChromaDB yet.
        return
    if not collection.metadata:
        # No metadata in the collection.
        return
    try:
        collection_model_type = collection.metadata["model_type"]
    except KeyError:
        ...  # No model type in the metadata.
    else:
        # Assume models from Azure and OpenAI with the same name as same.
        if collection_model_type != model_type and {
            model_type,
            collection_model_type,
        } != {"azure", "openai"}:
            raise RuntimeError(
                f"Given embeddings model type '{model_type}' doesn't "
                f"match with the embeddings model type "
                f"'{collection_model_type}' of the "
                f"collection '{collection_name}' of the ChromaDB."
            )
    try:
        collection_model_name = collection.metadata["model_name"]
    except KeyError:
        ...  # No model name in the metadata.
    else:
        if collection_model_name != model_name:
            raise RuntimeError(
                f"Given embeddings model name '{model_name}' doesn't "
                f"match with the embeddings model name "
                f"'{collection_model_name}' of the "
                f"collection '{collection_name}' of the database."
            )


def get_chromadb(
    embeddings_model: Optional[Embeddings] = None,
    persist_directory: Optional[Path] = None,
    collection_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
    relevance_score_fn: RelevanceScoreFn = "l2",
) -> Chroma:
    """
    https://docs.trychroma.com/usage-guide#changing-the-distance-function
    """
    if embeddings_model is not None and persist_directory is not None:
        _assert_embeddings_model_ok_for_chromadb(
            embeddings_model, persist_directory, collection_name
        )

    kwargs: Dict = {
        "collection_name": collection_name,
        "embedding_function": embeddings_model,
        "persist_directory": (
            None if persist_directory is None else str(persist_directory)
        ),
    }
    collection_metadata = {}
    if embeddings_model is not None:
        model_name, model_type = get_embeddings_model_config(embeddings_model)
        collection_metadata["model_name"] = model_name
        collection_metadata["model_type"] = model_type

    if isinstance(relevance_score_fn, str):
        assert relevance_score_fn in get_args(PredefinedRelevanceScoreFn)
        collection_metadata["hnsw:space"] = relevance_score_fn
    else:
        kwargs["relevance_score_fn"] = relevance_score_fn
    kwargs["collection_metadata"] = collection_metadata
    return Chroma(**kwargs)


def get_retriever(
    vectorstore: VectorStore,
    k: Optional[int] = None,
    search_type: RetrieverSearchType = "similarity",
    score_threshold: Optional[float] = None,
    fetch_k: Optional[int] = None,
    lambda_mult: Optional[float] = None,
) -> VectorStoreRetriever:
    kwargs: Dict[str, Any] = {"search_type": search_type}
    search_kwargs: Dict[str, Any] = {}
    if k is not None:
        search_kwargs["k"] = k
    if search_type == "similarity_score_threshold":
        assert score_threshold is not None
        search_kwargs["score_threshold"] = score_threshold
    elif search_type == "mmr":
        if fetch_k is not None:
            search_kwargs["fetch_k"] = fetch_k
        if lambda_mult is not None:
            search_kwargs["lambda_mult"] = lambda_mult
    kwargs["search_kwargs"] = search_kwargs
    return vectorstore.as_retriever(**kwargs)


def format_doc(doc: Document) -> str:
    return f"Content: {doc.page_content}\nSource: \"{doc.metadata['source']}\""


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(map(format_doc, docs))


def get_rag_chain(retriever: VectorStoreRetriever, llm: LLM) -> Runnable:
    template = """You are an assistant for question-answering tasks.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}
=========
{source_documents}
=========
FINAL ANSWER: """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            source_documents=(lambda x: format_docs(x["source_documents"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {  # type: ignore
            "source_documents": retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)
    return rag_chain_with_source


def get_rag_chain_with_task_breakdown(retriever: VectorStoreRetriever, llm: LLM) -> Runnable:
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    breakdown_prompt = PromptTemplate(
        template="""
            You are an expert in solid state battery, and your task is to answer user's question professionally.
            You need to design several intermediate steps to enhance processing and reasoning, aiming to reach the final answers through the intermediate steps.
            Ensure that each intermediate step is interconnected and collectively addresses the user's inquiry. These intermediate thoughts should encompass various aspects of solid state battery, requiring information retrieval from a comprehensive library.
            Your approach should systematically break down the query, ensuring a thorough exploration of the topic from multiple angles to provide a well-rounded answer.
            Your steps needs to be closely involve different aspects in solid state battery and material science, you must contain domain knowledge aspects in each step, and include reasoning thoughts
            You only need to list the intermeditae steps.
            Your output should be in unified format, for each step, encapsulate it as
            [step 1]
            the content for step1
            [step 2]
            the content for step2
            [step 3]
            the content for step3
            ...... \n\n
            Here is the user question: {question} \n\n
            Here is the context for the user question: \n\n {source_documents} \n
            The context for the user's question is retrieved from the literature database, which contains papers on solid-state batteries, using similarity search. \n
            """,
        input_variables=["question", "source_documents"],
    )

    step_analysis_prompt = PromptTemplate(
        # - When data visualization is necessary, generate Python code using libraries such as matplotlib, seaborn, or plotly to create visual representations that help explain your thoughts. Generating the Python code is sufficient; do not attempt to execute it.\n
        template="""
            You are an expert in solid state battery, and your task is to answer user's question professionally:\n
                - Informative: Furnish detailed insights, drawing upon the most relevant facts and figures.\n
                - Correct: Ensure factual accuracy in every aspect of your response.\n
                - Knowledgeable: Display a profound understanding of the subject matter, including advanced concepts and recent advancements in the field of solid state battery.\n
                - Holistic: Offer a well-rounded perspective, considering various facets of the topic at hand.\n\n
            You are given a list of interconnected intermediate steps toward answering a question on solid state battery. As you address each question, please adhere to the following guidelines:\n
                - Stay On Topic: Concentrate solely on the query posed. Your reply should be closely aligned with the question, avoiding tangential or unrelated content.\n
                - Analysis of Intermediate Thoughts: Make sure to follow these intermediate steps sequentially and address each step in a comprehensive and coherent manner, providing detailed explanations and insights.\n
                - Before responding, take a moment to center yourself. Breathe deeply, and proceed with a step-by-step analytical approach, ensuring that your expertise shines through in a manner that is both engaging and enlightening.\n
                - Data Visualization: you are good at data visualization and know how to use the Python library, matplotlib, to create visual representations that help explain your thoughts. When generating the Python visualization code, please adhere to the following instructions:\n
                    * Ensure that the Python visualization code is bug-free. Expecially, make sure that all variables are defined before they are used in the generated code.\n
                    * Use the subplots() method to create a figure object, fig, and an Axes object, ax. And then use the Axes object, ax, for writing the code of actual plotting. \n
                    * Do not use matplotlib.pyplot to directly write the code for plotting. Instead, use the the Axes object, ax. \n
                    * Do not add a piece of code to show the plot.\n
                    * Before outputing the code, ensure the input data for the plot is well defined.\n
                    * Generating the Python code is sufficient; do not attempt to execute it.\n
            ...... \n\n
            Here is the user question: {question} \n\n
            Here is the list of intermediate steps: \n\n {steps} \n\n
            Here is the context for the user question: \n\n {source_documents} \n
            The context for the user's question is retrieved from the literature database, which contains papers on solid-state batteries, using similarity search. \n
            """,
        input_variables=["question", "source_documents", "steps"],
    )
    breakdown_analysis_parser = breakdown_prompt | llm | StrOutputParser()
    breakdown_analysis = (
        RunnablePassthrough.assign(
            source_documents=(lambda x: format_docs(x["source_documents"]))
        )
        | { "question": RunnablePassthrough(), "source_documents": RunnablePassthrough(), "steps": breakdown_analysis_parser}
        | step_analysis_prompt
        | llm
        | StrOutputParser()
    )

    answer_with_intermediate_steps = RunnableParallel(
        {  # type: ignore
            "source_documents": retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=breakdown_analysis)
    return answer_with_intermediate_steps


def stable_hash(doc: Document) -> str:
    """
    Stable hash document based on its metadata.
    """
    return hashlib.sha1(json.dumps(doc.metadata, sort_keys=True).encode()).hexdigest()


def question_as_doc(question: str, rag_answer: Dict[str, Any]) -> Document:
    return Document(
        page_content=question,
        metadata={
            "answer": rag_answer["answer"],
            "sources": ",".join(map(stable_hash, rag_answer["source_documents"])),
        },
    )
