import torch
from tensorus.models.fedavg_model import FedAvgModel
from tensorus.models.large_language_model import LargeLanguageModelWrapper


def test_fedavg_aggregation():
    model = torch.nn.Linear(2, 1)
    client_state = [{k: torch.ones_like(v) for k, v in model.state_dict().items()} for _ in range(2)]
    agg = FedAvgModel(model)
    agg.fit(client_state)
    for p in agg.global_model.parameters():
        assert torch.allclose(p, torch.ones_like(p))


def test_large_language_model_wrapper(tmp_path):
    wrapper = LargeLanguageModelWrapper(model_name="hf-internal-testing/tiny-random-gpt2", epochs=1)
    wrapper.tokenizer.pad_token = wrapper.tokenizer.eos_token
    out = wrapper.generate(["hello"], max_length=5)[0]
    assert isinstance(out, str)
