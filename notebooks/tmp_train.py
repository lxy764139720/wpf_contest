import time
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from notebooks.tmp_lstm import LSTMModel
from torchinfo import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(dataloader, net, loss_fn, optimizer, batch_size, sample_size):
    # start = time.time()
    train_loss = []
    size = int(len(dataloader.dataset) * sample_size)
    net.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = net(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, cur = loss.item() / batch_size, batch * len(y)
        train_loss.append(loss)
        print(f"loss: {loss:>7f}  [{cur:>6d}/{size:>6d}]")
        # print("train %s" % (timeSince(start, cur / size)))
    return train_loss


def train_epochs(train_dataloader, net, loss_fn, optimizer, writer, num_epochs, batch_size, sample_size):
    train_ls = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_ls_epoch = train_epoch(train_dataloader, net, loss_fn, optimizer, batch_size, sample_size)
        train_ls_avg = np.mean(train_ls_epoch)
        print(f'Epoch {epoch + 1}: train loss: {train_ls_avg}')

        writer.add_scalar(f"train loss", train_ls_avg, epoch)
        train_ls.append(train_ls_avg)
    return train_ls


def train(log_dir, model_path, dataset, num_epochs=100, learning_rate=0.001, weight_decay=0.2, batch_size=32,
          sample_size=0.01):
    train_sampler = RandomSampler(data_source=dataset, num_samples=int(sample_size * len(dataset)),
                                  replacement=True)
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)

    net = LSTMModel().to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss(reduction='sum').to(device)
    writer = SummaryWriter(log_dir)

    for train_X, train_y in train_dataloader:
        print("Shape of test X: ", train_X.shape)
        print("Shape of test y: ", train_y.shape, train_y.dtype)
        break
    summary(net, (batch_size, 288, 1), col_names=["input_size", "kernel_size", "output_size"], verbose=2)

    train_ls = train_epochs(train_dataloader, net, loss_fn, optimizer, writer, num_epochs, batch_size, sample_size)
    torch.save(net.state_dict(), model_path + '.pt')
    return train_ls
