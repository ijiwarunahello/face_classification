import torch
import torchvision.transforms as transforms
import argparse
from my_dataset import Mydataset
from network import Net
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt


def parser():
    parser = argparse.ArgumentParser(description="Pytorch_Perfume")
    parser.add_argument("--epochs", "--e", type=int, default=2, help="number of epochs to train (defaults: 2)")
    parser.add_argument("--lr", "--l", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--save-model", action="store_true", default=False, help="For saving the current model")
    args = parser.parse_args()
    return args


def train(args, model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 5 == 4:
            print("[%d, %5d] loss: %.3f" % (epoch, batch_idx + 1, running_loss / 5))
            train_loss = running_loss / 5
            running_loss = 0.0
    return train_loss


def test(args, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print("¥nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)¥n".format(
        test_loss, correct, len(test_loader.dataset), 100. * accuracy
    ))
    return test_loss, accuracy


def main():
    args = parser()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = Mydataset(root="./train_data", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = Mydataset(root="./test_data", transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)

    model = Net()
    summary(model, (3, 64, 64))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    x_epoch_data = []
    y_train_loss_data = []
    y_test_loss_data = []
    y_test_accuracy_data = []
    for epoch in range(1, args.epochs + 1):
        train_loss_per_epoch = train(args, model, trainloader, criterion, optimizer, epoch)
        test_loss_per_epoch, test_accuracy_per_epoch = test(args, model, testloader, criterion)

        x_epoch_data.append(epoch)
        y_train_loss_data.append(train_loss_per_epoch)
        y_test_loss_data.append(test_loss_per_epoch)
        y_test_accuracy_data.append(test_accuracy_per_epoch)

    plt.plot(x_epoch_data, y_train_loss_data, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(x_epoch_data, y_test_loss_data, label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(x_epoch_data, y_test_accuracy_data, label="test_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.show()

    if (args.save_model):
        torch.save(model.state_dict(), "perfume_cnn.pt")


if __name__ == '__main__':
    main()
