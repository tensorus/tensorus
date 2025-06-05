import torch

from tensorus.models.transformer_models import (
    TransformerModel,
    BERTModel,
    GPTModel,
    T5Model,
    VisionTransformerModel,
)


def test_transformer_forward():
    model = TransformerModel(input_dim=10, model_dim=8, num_heads=2)
    src = torch.randint(0, 10, (2, 5))
    out = model.predict(src)
    assert out.shape == (2, 5)


def test_bert_forward():
    model = BERTModel(num_classes=2, pretrained=False)
    x = torch.randint(0, model.model.config.vocab_size, (2, 4))
    preds = model.predict(x)
    assert preds.shape == (2,)


def test_gpt_forward():
    model = GPTModel(pretrained=False)
    x = torch.randint(0, model.model.config.vocab_size, (1, 4))
    seq = model.predict(x, max_length=5)
    assert seq.shape[0] == 1


def test_t5_forward():
    model = T5Model(pretrained=False)
    x = torch.randint(0, model.model.config.vocab_size, (1, 4))
    seq = model.predict(x, max_length=5)
    assert seq.shape[0] == 1


def test_vit_forward():
    model = VisionTransformerModel(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    preds = model.predict(x)
    assert preds.shape[0] == 1
