import torch
import torchvision
import torchvision.transforms as transforms

from models import MLPConvNet, ConvNet, ResNet, BasicBlock, Bottleneck

from optim import GradientDescent, AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 5
num_classes = 10
batch_size = 128
learning_rate = 0.001


def normalize(data_tensor):
    '''re-scale image values to [-1, 1]'''
    return (data_tensor / 255.) * 2. - 1.


def tile_image(image):
    '''duplicate along channel axis'''
    return image.repeat(3, 1, 1)


transform_list = [transforms.ToTensor(
), transforms.Lambda(lambda x: normalize(x))]

train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.Compose(transform_list+[
                                               transforms.ToPILImage(),
                                               transforms.Resize(32),
                                               transforms.ToTensor(),
                                               transforms.Lambda(lambda x: tile_image(x)), ]),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.Compose(transform_list+[
                                              transforms.ToPILImage(),
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              transforms.Lambda(
                                                  lambda x: tile_image(x)),
                                              # transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

mean = train_dataset.data.float().mean() / 255
std = train_dataset.data.float().std() / 255
print(f'Calculated mean: {mean}')
print(f'Calculated std: {std}')

print(f'Number of training examples: {len(train_dataset)}')
print(f'Number of testing examples: {len(test_dataset)}')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#model = ConvNet(num_classes).to(device)
#model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
print(model)

a

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = GradientDescent(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
losses = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(
        100 * correct / total))
