import torch

import logging


class Tester(object):
    def __init__(self,
                 model,
                 test_dataloader=None,
                 use_gpu=True
                 ) -> None:
        self.model = model
        self.test_dataloader = test_dataloader
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
            self.model = self.model.cuda()

    def run(self):
        correct = 0
        total = 0
        for images, labels in self.test_dataloader:
            if self.use_gpu:
                images = images.to(self.device)
                labels = labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        logging.info('Accuracy of the network on the 10000 test images: {:.5f} '.format(
            correct / total))
