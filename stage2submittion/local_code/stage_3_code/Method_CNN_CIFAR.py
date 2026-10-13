'''
Concrete MethodModule class: CNN for the CIFAR color object images (3 x 32 x 32, 10 classes).
All settings can be overridden through keyword arguments for the ablation study.
'''

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    def __init__(self, mName, mDescription, **kwargs):
        defaults = dict(in_channels=3, input_size=(32, 32), n_classes=10,
                        conv_channels=(32, 64), kernel_size=3, padding=1, stride=1,
                        pool='max', pool_size=2, hidden_dim=128, dropout=0.25,
                        learning_rate=1e-3, max_epoch=10, batch_size=128)
        defaults.update(kwargs)
        super().__init__(mName, mDescription, **defaults)