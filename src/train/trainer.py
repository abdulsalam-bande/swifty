from utils.swift_dock_logger import swift_dock_logger
logger = swift_dock_logger()


def train_model(train_dataloader, model, criterion, optimizer, number_of_epochs):
    metrics_dict = {"training_mse": []}
    mse_array_train = []
    for epoch in range(number_of_epochs):
        running_loss = 0
        for i_batch, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            features, target = data
            features = features.squeeze()
            features = features.unsqueeze(0)
            outputs = model(features).double()
            outputs = outputs.squeeze(0)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            del features, outputs
            running_loss += loss.item()
        mse_array_train.append(running_loss / len(train_dataloader))
        logger.info(f"Epoch: {epoch+1}/{number_of_epochs}")
        logger.info(f"Training mse {(running_loss / len(train_dataloader))}")

    metrics_dict["training_mse"] = mse_array_train
    logger.info("Finished training.")
    return model, metrics_dict


