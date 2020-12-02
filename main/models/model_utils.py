import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report


def run_batch(model, batch_data, criterion, optimizer=None):
    epoch_loss = 0

    for _, batch in enumerate(batch_data):
        source = batch['morphs']
        target = batch['entities']

        if optimizer:
            optimizer.zero_grad()

        # output: [batch_size, seq_len, num_entities]
        # target: [batch_size, seq_len]
        output = model(source, target)
        output_dim = output.shape[-1]

        # output: [batch_size * (seq_len - 1), num_entities]
        # target: [batch_size * (seq_len - 1)]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)

        loss = criterion(output, target)

        if optimizer:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(batch_data)


def train(model, train_data, test_data, num_epochs=50, test_step=5, learning_rate=0.005):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = run_batch(model, train_data, criterion, optimizer)
        validation_loss = float('inf')

        # Test via validation
        if (epoch + 1) % test_step == 0:
            validation_loss = evaluate(model, test_data, criterion)
            if validation_loss < best_valid_loss:
                best_valid_loss = validation_loss
                torch.save(model, 'savepoint.model')

        print('Epoch: {:02}/{:02} | Train Loss: {}'.format(epoch + 1, num_epochs, train_loss))

        if (epoch + 1) % test_step == 0:
            print('Test Step {:02}/{:02} | Validation Loss: {}'.format(
                int((epoch + 1) / test_step), test_step, validation_loss), end='\n\n')


def evaluate(model, test_data, criterion):
    model.eval()

    with torch.no_grad():
        loss = run_batch(model, test_data, criterion)

    return loss


def test(model, test_data):
    test_y = None
    predictions = None

    model.eval()
    with torch.no_grad():
        # TODO: Needs implemation (데이터형식을 아직 몰라서 남겨둠)
        pass

    return classification_report(test_y, predictions, zero_division='1')
