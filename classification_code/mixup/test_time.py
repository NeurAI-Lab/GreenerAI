import torch
import time

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(6 * 6 * 512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 6 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    net = Net()
    net = net.cuda()
    import torch.optim as optim
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # with torch.cuda.profiler.profile():
    #     with torch.autograd.profiler.emit_nvtx() as prof:
        for i, data in enumerate(trainloader):

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            net(inputs)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i > 1000:
                print("exit")
                break

    # NOTE: some columns were removed for brevity
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print(prof.total_average())
    # print(prof)
    # prof.export_chrome_trace("prof.json")

    # with torch.cuda.profiler.profile():
    #     x = torch.randn((1, 1), requires_grad=True)
    #     with torch.autograd.profiler.emit_nvtx():
    #         y = x ** 2
    #         y.backward()


if __name__ == '__main__':
    main()
