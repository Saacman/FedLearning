import torch
import torch.nn as nn
import torch.optim as optim

def qtrain_model(model, train_loader, device, criterion = None, optimizer = None, num_epochs=20, learning_rate=1e-2, momentum=0.9, weight_decay=1e-5):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Training
    model.to(device)
    model.train()
    for epoch in range(num_epochs):


        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        print(f"Epoch: {epoch}/{num_epochs} Train Loss: {train_loss:.3f} Train Acc: {train_accuracy:.3f}")

    return model