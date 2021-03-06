 import numpy as np
        import torch
        import mlflow.pytorch

        class LinearNNModel(torch.nn.Module):
            def __init__(self):
                super(LinearNNModel, self).__init__()
                self.linear = torch.nn.Linear(1, 1)  # One in and one out

            def forward(self, x):
                y_pred = self.linear(x)
                return y_pred

        def gen_data():
            # Example linear model modified to use y = 2x
            # from https://github.com/hunkim/PyTorchZeroToAll
            # X training data, y labels
            X = torch.arange(1.0, 25.0).view(-1, 1)
            y = torch.from_numpy(np.array([x * 2 for x in X])).view(-1, 1)
            return X, y

        # Define model, loss, and optimizer
        model = LinearNNModel()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Training loop
        epochs = 250
        X, y = gen_data()
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing X to the model
            y_pred = model(X)

            # Compute the loss
            loss = criterion(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

            # convert to scripted model and log the model
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

        # Fetch the logged model artifacts
        print("run_id: {}".format(run.info.run_id))
        for artifact_path in ["model/data", "scripted_model/data"]:
            artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id,
                        artifact_path)]
            print("artifacts: {}".format(artifacts))