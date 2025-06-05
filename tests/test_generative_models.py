import torch
import torchvision

from tensorus.models.vae_model import VAEModel
from tensorus.models.gan_model import GANModel
from tensorus.models.diffusion_model import DiffusionModel
from tensorus.models.flow_based_model import FlowBasedModel


def _get_fake_data(n: int = 20):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
    ])
    dataset = torchvision.datasets.FakeData(size=n, image_size=(1, 28, 28), transform=transform)
    X = torch.stack([dataset[i][0] for i in range(n)])
    return X


def test_vae_training_and_sampling():
    X = _get_fake_data()
    model = VAEModel(epochs=1, batch_size=10)
    model.fit(X)
    samples = model.sample(2)
    assert samples.shape[0] == 2


def test_gan_training_and_sampling():
    X = _get_fake_data()
    model = GANModel(epochs=1, batch_size=10)
    model.fit(X)
    samples = model.sample(2)
    assert samples.shape[0] == 2


def test_diffusion_training_and_sampling():
    X = _get_fake_data()
    model = DiffusionModel(epochs=1, batch_size=10, steps=2)
    model.fit(X)
    samples = model.sample(2)
    assert samples.shape[0] == 2


def test_flow_based_training_and_sampling():
    X = _get_fake_data()
    model = FlowBasedModel(epochs=1, batch_size=10)
    model.fit(X)
    samples = model.sample(2)
    assert samples.shape[0] == 2
