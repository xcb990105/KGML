import torch
import pandas as pd
from utils.cnn_preprocess import prepare_dataloader
from models.delta_G.cnn_network import CNNResnet50
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim import Adam
from sklearn.metrics import r2_score, mean_squared_error
import glob


class CNNDeltaGModel:

    def __init__(self, train_path : str, val_path : str, test_path : str, image_path : str,ckpt_path : str, log_path : str,
                 out_channels : int, hidden_channels : int, metal_feature_dim=8,
                 metal_embed_dim=128, lr=1e-4, bs=32, model_path=None, network=CNNResnet50):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.image_path = image_path
        self.ckpt_path = ckpt_path
        self.log_path = log_path
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.metal_feature_dim = metal_feature_dim
        self.metal_embed_dim = metal_embed_dim
        self.lr = lr
        self.bs = bs
        self.train_loader = prepare_dataloader(self.train_path, self.image_path, True, self.bs)
        self.val_loader = prepare_dataloader(self.val_path, self.image_path, False, self.bs)
        self.test_loader = prepare_dataloader(self.test_path, self.image_path, False, self.bs)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = network(hidden_channels=self.hidden_channels,
                                       out_channels=self.out_channels,
                                       metal_feature_dim=self.metal_feature_dim,
                                       metal_embed_dim=self.metal_embed_dim).to(self.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def fit(self, num_epochs=100):
        os.makedirs(self.log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_path)
        os.makedirs(self.ckpt_path, exist_ok=True)
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        cur_epoch = -1
        min_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for data, batch_metal_features, target, _ in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)
                optimizer.zero_grad()

                output = self.model(data, batch_metal_features)
                if isinstance(output, tuple): 
                    prediction, _ = output
                else:
                    prediction = output
                loss = loss_fn(prediction, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, batch_metal_features, target, _ in self.val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    batch_metal_features = batch_metal_features.to(self.device)
                    output = self.model(data, batch_metal_features)
                    if isinstance(output, tuple):
                        prediction, _ = output
                    else:
                        prediction = output
                    loss = loss_fn(prediction, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                cur_epoch = epoch
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)

            checkpoint_path = os.path.join(self.ckpt_path, f'model_epoch_{epoch + 1}.pt')
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_val_loss:.6f}")
        print(f"The best performing epoch is {cur_epoch + 1}.")
        print(f"Load epoch {cur_epoch + 1} model parameters")
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, f'model_epoch_{cur_epoch + 1}.pt'), map_location=self.device))
        self.load_and_clean_checkpoints(cur_epoch)

    def load_and_clean_checkpoints(self, cur_epoch):

        current_checkpoint = os.path.join(self.ckpt_path, f'model_epoch_{cur_epoch + 1}.pt')

        self.model.load_state_dict(
            torch.load(current_checkpoint, map_location=self.device)
        )

        all_checkpoints = glob.glob(os.path.join(self.ckpt_path, 'model_epoch_*.pt'))

        for checkpoint in all_checkpoints:
            if checkpoint != current_checkpoint:
                try:
                    os.remove(checkpoint)
                    # print(f"Deleted old checkpoint: {checkpoint}")
                except Exception as e:
                    print(f"Failed to delete {checkpoint}: {str(e)}")

    def predict_train_test(self, result_train_path=None, result_val_path=None, result_test_path=None):
        self.model.eval()
        results_test = []
        results_val = []
        results_train = []
        # model.load_state_dict(torch.load("checkpoints/gcn_deltaG_prediction_01/model_epoch_30.pt", map_location=device))
        with torch.no_grad():
            for data, batch_metal_features, target, smiles_batch in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)

                output = self.model(data, batch_metal_features)
                predictions = output.squeeze(-1).cpu().numpy()

                labels = target.cpu().numpy()

                for smi, pred, label in zip(smiles_batch, predictions, labels):
                    results_test.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label
                    })

            for data, batch_metal_features, target, smiles_batch in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)

                output = self.model(data, batch_metal_features)
                predictions = output.squeeze(-1).cpu().numpy()

                labels = target.cpu().numpy()

                for smi, pred, label in zip(smiles_batch, predictions, labels):
                    results_val.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label
                    })

            for data, batch_metal_features, target, smiles_batch in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)
                output = self.model(data, batch_metal_features)
                predictions = output.squeeze(-1).cpu().numpy()

                labels = target.cpu().numpy()

                for smi, pred, label in zip(smiles_batch, predictions, labels):
                    results_train.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label
                    })

        results_test_df = pd.DataFrame(results_test)
        results_val_df = pd.DataFrame(results_val)
        results_train_df = pd.DataFrame(results_train)
        if result_test_path is not None:
            results_test_df.to_csv(result_test_path, index=False)
        if result_train_path is not None:
            results_train_df.to_csv(result_train_path, index=False)
        if result_val_path is not None:
            results_val_df.to_csv(result_val_path, index=False)
        # target_pred_y = scaler.inverse_transform(predicted_y).values.reshape(-1, 1))
        print("Train R2:", r2_score(results_train_df['true_value'], results_train_df['predicted_value']))
        print("Val R2:", r2_score(results_val_df['true_value'], results_val_df['predicted_value']))
        print("Test R2:", r2_score(results_test_df['true_value'], results_test_df['predicted_value']))
        print("Train mse:", mean_squared_error(results_train_df['true_value'], results_train_df['predicted_value']))
        print("Val mse:", mean_squared_error(results_val_df['true_value'], results_val_df['predicted_value']))
        print("Test mse:", mean_squared_error(results_test_df['true_value'], results_test_df['predicted_value']))
        return ((r2_score(results_train_df['true_value'], results_train_df['predicted_value']),
                 r2_score(results_val_df['true_value'], results_val_df['predicted_value']),
                 r2_score(results_test_df['true_value'], results_test_df['predicted_value'])),
                (mean_squared_error(results_train_df['true_value'], results_train_df['predicted_value']),
                 mean_squared_error(results_val_df['true_value'], results_val_df['predicted_value']),
                 mean_squared_error(results_test_df['true_value'], results_test_df['predicted_value'])),)