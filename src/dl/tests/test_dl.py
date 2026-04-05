"""
PrajnaAI — Deep Learning Tests (CPU PyTorch, no GPU required)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from driver import LSTMClassifier, SentimentDataset, TabularMLP, train_lstm_classifier


class TestSentimentDataset:

    @pytest.fixture
    def dataset(self):
        return SentimentDataset()

    def test_dataset_size(self, dataset):
        assert len(dataset) == 24

    def test_encoding_length(self, dataset):
        x, y = dataset[0]
        assert x.shape[0] == dataset.max_len

    def test_labels_binary(self, dataset):
        for i in range(len(dataset)):
            _, y = dataset[i]
            assert y.item() in [0.0, 1.0]

    def test_vocab_has_pad(self, dataset):
        assert "<pad>" in dataset.vocab
        assert dataset.vocab["<pad>"] == 0

    def test_encode_pads_correctly(self, dataset):
        short_text = "good"
        encoded = dataset.encode(short_text)
        assert len(encoded) == dataset.max_len
        # Should have padding zeros at the end
        assert 0 in encoded


class TestLSTMClassifier:

    @pytest.fixture
    def model(self):
        dataset = SentimentDataset()
        return LSTMClassifier(vocab_size=len(dataset.vocab))

    def test_forward_shape(self, model):
        x = torch.randint(0, 50, (4, 15))  # batch=4, seq_len=15
        output = model(x)
        assert output.shape == (4,)

    def test_output_range(self, model):
        x = torch.randint(0, 50, (8, 15))
        with torch.no_grad():
            output = model(x)
        assert (output >= 0).all() and (output <= 1).all()

    def test_parameter_count(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 100_000, f"Model too large for CPU demo: {total_params:,} params"

    def test_gradient_flow(self, model):
        x = torch.randint(0, 50, (4, 15))
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = torch.nn.BCELoss()(model(x), y)
        loss.backward()
        # Check at least one gradient is non-zero
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters())
        assert has_grad

    @pytest.mark.slow
    def test_training_improves(self):
        result = train_lstm_classifier(epochs=20)
        # Loss should decrease over training
        first_loss = result["history"]["loss"][0]
        last_loss = result["history"]["loss"][-1]
        assert last_loss < first_loss, "Training loss should decrease"

    @pytest.mark.slow
    def test_final_accuracy_threshold(self):
        result = train_lstm_classifier(epochs=30)
        final_acc = result["history"]["accuracy"][-1]
        assert final_acc > 0.7, f"Final accuracy {final_acc:.3f} too low"


class TestTabularMLP:

    @pytest.fixture
    def model(self):
        return TabularMLP(input_dim=8)

    def test_forward_shape(self, model):
        x = torch.randn(16, 8)
        output = model(x)
        assert output.shape == (16,)

    def test_batch_norm_eval_mode(self, model):
        model.eval()
        x = torch.randn(1, 8)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1,)

    def test_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total < 50_000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
