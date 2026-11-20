'''
Concrete SettingModule class: the train/test sets are already partitioned into two files,
so no random split or cross validation is needed.
'''

# Based on the ECS 170 code template (Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>)
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test(setting):
    test_dataset = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestDataset=None):
        super().prepare(sDataset, sMethod, sResult, sEvaluate)
        self.test_dataset = sTestDataset

    def load_run_save_evaluate(self):
        # load the pre-partitioned training and testing sets
        train_data = self.dataset.load()
        test_data = self.test_dataset.load()

        # run MethodModule
        self.method.data = {'train': train_data, 'test': test_data}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None
