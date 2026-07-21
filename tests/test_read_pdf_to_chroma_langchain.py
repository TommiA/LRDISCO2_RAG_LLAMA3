import pytest
from unittest.mock import MagicMock, patch

import read_pdf_to_chroma_langchain as app


def test_read_pdf_unstructured_elements_calls_partition_pdf():
    mock_partition_pdf = MagicMock(return_value=["element"])
    with patch("unstructured.partition.pdf.partition_pdf", mock_partition_pdf):
        elements = app.read_pdf_unstructured_elements("data/pdf/test.pdf")

    assert elements == ["element"]
    mock_partition_pdf.assert_called_once()


def test_split_doc_splits_documents():
    fake_document = MagicMock(page_content="hello")
    fake_splitter = MagicMock()
    fake_splitter.split_documents.return_value = ["split"]

    with patch("read_pdf_to_chroma_langchain.RecursiveCharacterTextSplitter", return_value=fake_splitter):
        splits = app.split_doc([fake_document])

    assert splits == ["split"]
    fake_splitter.split_documents.assert_called_once_with([fake_document])


def test_store_basic_docs_creates_collection_and_adds():
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_collection
    docs = [MagicMock(page_content="page1"), MagicMock(page_content="page2")]

    with patch.object(app, 'client', mock_client):
        app.store_basic_docs(docs)

    mock_client.create_collection.assert_called_once_with(name="LR_Disco_2_docs")
    assert mock_collection.add.call_count == 2


def test_nomic_embed_and_store_uses_embed_text():
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_collection.get.return_value = {'ids': ['1'], 'documents': ['doc']}
    mock_output_collection = MagicMock()
    mock_client.create_collection.return_value = mock_output_collection

    with patch("read_pdf_to_chroma_langchain.embed.text", return_value={"embeddings": [[0.1, 0.2]]}) as mock_embed_text:
        app.nomic_embed_and_store(mock_client)

    mock_embed_text.assert_called_once()
    mock_output_collection.add.assert_called_once()


def test_embed4all_embed_and_store_uses_Embed4All():
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_collection.get.return_value = {'ids': ['1'], 'documents': ['doc']}
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2]
    mock_output_collection = MagicMock()
    mock_client.create_collection.return_value = mock_output_collection

    with patch("read_pdf_to_chroma_langchain.Embed4All", return_value=mock_embedder):
        app.embed4all_embed_and_store(mock_client)

    mock_embedder.embed.assert_called_once_with('doc')
    mock_output_collection.add.assert_called_once()


def test_hf_embed_and_store_uses_huggingface():
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_collection.get.return_value = {'ids': ['1'], 'documents': ['doc']}
    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1, 0.2]]
    mock_output_collection = MagicMock()
    mock_client.create_collection.return_value = mock_output_collection

    with patch("read_pdf_to_chroma_langchain.HuggingFaceEmbeddings", return_value=mock_embedder):
        app.hf_embed_and_store(mock_client)

    mock_embedder.embed_documents.assert_called_once()
    mock_output_collection.add.assert_called_once()
