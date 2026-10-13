'''
Concrete MethodModule class: CNN for the ORL face images (1 x 112 x 92, 40 classes).
All settings can be overridden through keyword arguments for the ablation study.
'''

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    def __init__(self, mName, mDescription, **kwargs):
        defaults = dict(in_channels=1, input_size=(112, 92), n_classes=40,
                        conv_channels=(32, 64), kernel_size=3, padding=1, stride=1,
                        pool='max', pool_size=2, hidden_dim=128, dropout=0.5,
                        learning_rate=1e-3, max_epoch=30, batch_size=16)
        defaults.update(kwargs)
        super().__init__(mName, mDescription, **defaults)