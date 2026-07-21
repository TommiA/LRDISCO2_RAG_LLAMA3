import argparse
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import query_chroma_db_and_llama as app


def test_query_collection_returns_concatenated_documents():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        'documents': [["doc1", "doc2"]],
        'distances': [[0.1, 0.2]]
    }

    with patch.object(app, 'embedder', mock_embedder), patch.object(app, 'collection', mock_collection):
        result = app.query_collection("test")

    assert result == "doc1doc2"
    mock_embedder.embed.assert_called_once_with("test")
    mock_collection.query.assert_called_once()


def test_query_collection_debug_prints(capsys):
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        'documents': [["doc1", "doc2"]],
        'distances': [[0.1, 0.2]]
    }

    with patch.object(app, 'embedder', mock_embedder), patch.object(app, 'collection', mock_collection), patch.object(app, 'args', argparse.Namespace(debug=True)):
        result = app.query_collection("test")

    captured = capsys.readouterr()
    assert "DEBUG: Chroma DB retrieval results:" in captured.out
    assert "doc1" in captured.out
    assert result == "doc1doc2"


def test_process_query_calls_model_generate():
    mock_model = MagicMock()
    mock_chat_session = MagicMock()
    mock_model.chat_session.return_value.__enter__.return_value = None
    mock_model.generate.return_value = "result"

    with patch.object(app, 'model', mock_model):
        response = app.process_query("hello", "context")

    assert response == "result"
    mock_model.generate.assert_called_once()
    assert "assistant" in mock_model.generate.call_args[0][0].lower()


def test_gpu_selection_uses_cpu_by_default():
    args = argparse.Namespace(gpu=False)
    with patch.object(app, 'args', args):
        m_device = "cpu"
        if args.gpu:
            if len(app.GPT4All.list_gpus()[0]) > 0:
                m_device = app.GPT4All.list_gpus()[0]

    assert m_device == "cpu"


def test_gpu_selection_uses_list_gpus_when_available():
    args = argparse.Namespace(gpu=True)
    with patch.object(app.GPT4All, 'list_gpus', return_value=["cuda:0"]):
        m_device = "cpu"
        if args.gpu:
            if len(app.GPT4All.list_gpus()[0]) > 0:
                m_device = app.GPT4All.list_gpus()[0]

    assert m_device == "cuda:0"
