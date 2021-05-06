import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pseudo_label.pseudo_label_classifier import PseudoLabelClassifier
from pseudo_label.data import PseudoLabelDataset, ModelLabeledDataset

from torch_tools.utils import wrap_with_tqdm
from torch_tools.visualization import show


def binary_classifier(train_dataloader, val_dataloader,
                      criterion=torch.nn.CrossEntropyLoss(), verbose=True, size=None):
    channels = next(iter(train_dataloader))[0].shape[1]
    model = PseudoLabelClassifier(channels=channels).cuda().train()
    opt = torch.optim.Adam(model.parameters())
    val_loss = 0.0
    accuracy = 0.0
    n_steps = 100
    val_steps = 10

    for step, sample in wrap_with_tqdm(enumerate(iter(train_dataloader)), verbose):
        if step >= n_steps:
            break

        imgs, labels = sample
        if size is not None:
            imgs = F.interpolate(imgs, size)
        model.zero_grad()

        prediction = model(imgs)
        loss = criterion(prediction, labels)
        loss.backward()
        opt.step()

        if step % 50 == 0 and verbose:
            print('step {}: {}'.format(step, loss.item()))

        if step == n_steps - 1:
            model.eval()
            val_loss, accuracy = validate(model, val_dataloader, val_steps, criterion)
            model.train()

            if verbose:
                print('Validation loss: {:.3} accuracy: {:.3}'.format(val_loss, accuracy))

    return model.eval(), val_loss, accuracy


def validate(model, dataloader, steps, criterion=torch.nn.CrossEntropyLoss(), size=None):
    with torch.no_grad():
        accuracy = 0.0
        val_loss = 0.0

        for val_step, sample in enumerate(dataloader):
            if val_step >= steps:
                break
            imgs, labels = sample
            if size is not None:
                imgs = F.interpolate(imgs, size)
            prediction = model(imgs)
            val_loss += criterion(prediction, labels).item()
            accuracy += torch.mean(
                (torch.argmax(prediction, dim=1) == labels).to(torch.float)).item()

    return val_loss / steps, accuracy / steps


def inspect_dims_transferability(G, deformator, dim, real_data_train, real_data_test, r, size=None,
                                 verbose=False):
    data = PseudoLabelDataset(G, dim, r=r, deformator=deformator, size=size, batch_size=32)
    val_data = PseudoLabelDataset(G, dim, r=r, deformator=deformator, size=size, batch_size=32)

    model, start_loss, start_accuracy = binary_classifier(
        data, val_data, verbose=verbose, size=size)

    real_data_train = ModelLabeledDataset(real_data_train, model)
    real_data_test = ModelLabeledDataset(real_data_test, model)

    model_by_real_data, real_data_loss, real_data_accuracy = binary_classifier(
        DataLoader(real_data_train, batch_size=32, shuffle=True),
        DataLoader(real_data_test, batch_size=32, shuffle=True),
        verbose=verbose, size=size)

    returned_loss, returned_accuracy = validate(model_by_real_data, val_data, 100, size=size)
    return start_accuracy, real_data_accuracy, returned_accuracy, model, model_by_real_data