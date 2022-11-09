import torch as t
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons

def _get_moon_data(unsqueeze_y=False):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = t.tensor(X, dtype=t.float32)
    y = t.tensor(y, dtype=t.int64)
    if unsqueeze_y:
        y = y.unsqueeze(-1)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

def _train_with_opt(model, opt):
    dl = _get_moon_data()
    for i, (X, y) in enumerate(dl):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(5, 3), nn.ReLU())
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.classifier(self.base(x))

def construct_param_config_from_description(description, model):
    param_config = []
    for param_group in description:
        param_group_ = param_group.copy()
        param_group_["params"] = getattr(model, param_group_["params"]).parameters()
        param_config.append(param_group_)
    return param_config

def test_sgd_param_groups(SGD):
    test_cases = [
        (
            [{'params': "base"}, {'params': "classifier", 'lr': 1e-3}],
            dict(lr=1e-2, momentum=0.0),
        ),
        (
            [{'params': "base"}, {'params': "classifier"}],
            dict(lr=1e-2, momentum=0.9),
        ),
        (
            [{'params': "base", "lr": 1e-2, "momentum": 0.95}, {'params': "classifier", 'lr': 1e-3}],
            dict(momentum=0.9, weight_decay=0.1),
        ),
    ]
    for description, kwargs in test_cases:
        t.manual_seed(819)

        model = Net()
        param_config = construct_param_config_from_description(description, model)
        opt = optim.SGD(param_config, **kwargs)
        _train_with_opt(model, opt)
        w0_correct = model.base[0].weight
        
        t.manual_seed(819)
        model = Net()
        param_config = construct_param_config_from_description(description, model)
        opt = SGD(param_config, **kwargs)
        _train_with_opt(model, opt)
        w0_submitted = model.base[0].weight

        print("\nTesting configuration: ", description)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)

    print("\nTesting that your function doesn't allow duplicates (this should raise an error): ")
    description, kwargs = (
        [{'params': "base", "lr": 1e-2, "momentum": 0.95}, {'params': "base", 'lr': 1e-3}],
        dict(momentum=0.9, weight_decay=0.1),
    )
    try:
        model = Net()
        param_config = construct_param_config_from_description(description, model)
        opt = SGD(param_config, **kwargs)
    except:
        print("Got an error, as expected.\n")
    else:
        raise Exception("Should have gotten an error from using duplicate parameters, but didn't.")
    

    print("All tests in `test_sgd_param_groups` passed!")











class SGD:

    def __init__(self, params, **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        kwargs can contain lr, momentum or weight_decay
        '''
        self.param_groups = []
        default_param_values = dict(momentum = 0., weight_decay = 0.)

        checkParams = set()

        for param_group in params:
            param_group = {**default_param_values, **kwargs, **param_group}
            param_group['params'] = list(param_group['params'])
            param_group['bs'] = [t.zeros_like(p) for p in param_group['params']]
            assert param_group['lr'] is not None, "lr must be specified"

            self.param_groups.append(param_group)

            for param in param_group['params']:
                assert param not in checkParams, "WHY?!"
                checkParams.add(param)


        
            
        self.t = 1

        
        # check if parameters appear in more than one group?




    @t.inference_mode()
    def step(self):

        pg: dict
        for i, pg in enumerate(self.param_groups):
            params = pg.get('params')
            assert params is not None, 'must have parameter'
            lr = pg['lr']          
            momentum = pg['momentum']
            weight_decay = pg['weight_decay']
                       
            for j, param in enumerate(params):
                g = param.grad
                if weight_decay != 0:
                    g = g + weight_decay * param
                if momentum != 0:
                    if self.t > 1:
                        b = momentum * pg.get('bs')[j] + g
                    else:
                        b = g
                    g = b
                    pg.get('bs')[j] = b
                param -= lr * g 
        self.t = self.t + 1

    def zero_grad(self) -> None:
        for param_group in self.param_groups:
            for param in param_group.get('params'):
                param.grad = None

utils.test_sgd_param_groups(SGD)