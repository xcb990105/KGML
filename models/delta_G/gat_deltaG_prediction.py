import torch
import pandas as pd
from utils.smiles2graph import weights_to_colormap
from utils.gnn_preprocess import prepare_dataloader, prepare_dataloader_PI, prepare_PI_model, test_dataloader
from models.delta_G.gat_network import GATCrossAttention
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim import Adam
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import  MinMaxScaler
import glob
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def nll_loss(mean, logvar, target):
    return 0.5 * (logvar + ((target - mean) ** 2) / torch.exp(logvar)).mean()

class GATDeltaGModel:
    cv_train_loader = None
    cv_val_loader = None
    last_bs = None
    def __init__(self, train_path : str, val_path : str, test_path : str, ckpt_path : str, log_path : str, in_channels : int,
                 out_channels : int, gat_hidden_channels : int, gat_layers : int, metal_feature_dim=8,
                 metal_embed_dim=128, query_dim=128, key_dim=128, value_dim=128, output_hidden_dim=512, lr=1e-4, bs=32, model_path=None,
                 network=GATCrossAttention, uncertainty=False, cv_path=None, PI=False, pretrain_path=None, ext=False):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.ckpt_path = ckpt_path
        self.log_path = log_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gat_hidden_channels = gat_hidden_channels
        self.gat_layers = gat_layers
        self.metal_feature_dim = metal_feature_dim
        self.metal_embed_dim = metal_embed_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_hidden_dim = output_hidden_dim
        self.lr = lr
        self.bs = int(bs)
        self.PI = PI
        self.pretrain_path = pretrain_path
        # self.val_loader = prepare_dataloader(self.val_path, False, self.bs)
        if cv_path is not None:
            if GATDeltaGModel.last_bs is None or GATDeltaGModel.last_bs != self.bs:
                GATDeltaGModel.last_bs = self.bs
                GATDeltaGModel.cv_train_loader = []
                GATDeltaGModel.cv_val_loader = []
                for i in range(1, 6):
                    GATDeltaGModel.cv_train_loader.append(prepare_dataloader(os.path.join(cv_path, f"fold_{i}_train.csv"), True, self.bs))
                    GATDeltaGModel.cv_val_loader.append(prepare_dataloader(os.path.join(cv_path, f"fold_{i}_val.csv"), False, self.bs))
        elif PI:
            scaler = MinMaxScaler()
            if ext:
                self.pi_loader = prepare_dataloader_PI("data/pi_A_des.csv", True, self.bs, scaler=scaler, fit=True, val_path=self.val_path, test_path=self.test_path)
            else:
                self.pi_loader = prepare_dataloader_PI("data/pi_A_des.csv", True, self.bs, scaler=scaler, fit=True, train_path=self.train_path)
            self.pi_model = prepare_PI_model(self.train_path, scaler=scaler)
            self.train_loader = prepare_dataloader(self.train_path, True, self.bs, scaler)
            self.val_loader = prepare_dataloader(self.val_path, False, self.bs, scaler)
            self.test_loader = prepare_dataloader(self.test_path, False, self.bs, scaler)
        else:
            self.train_loader = prepare_dataloader(self.train_path, True, self.bs)
            self.val_loader = prepare_dataloader(self.val_path, False, self.bs)
            self.test_loader = prepare_dataloader(self.test_path, False, self.bs)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.network = network
        self.model = self.network(node_in_channels=self.in_channels, hidden_channels=self.gat_hidden_channels,
                             out_channels=self.out_channels, num_node_layers=self.gat_layers,
                             metal_feature_dim=self.metal_feature_dim,
                             metal_embed_dim=self.metal_embed_dim, query_dim=self.query_dim,
                             key_dim=self.key_dim, value_dim=self.value_dim, output_hidden_dim=self.output_hidden_dim).to(self.device)
        self.uncertainty = uncertainty
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def internal_pi_log(self, dataloader, f_loader, loss_fn, loss_title, log_writer, epoch):
        delta_G_loss = 0
        F_loss = 0
        pi_loss = 0
        for data, batch_metal_features in dataloader:
            data = data.to(self.device)
            batch_metal_features = batch_metal_features.to(self.device)
            output, des = self.model(data, batch_metal_features)
            prediction = output
            loss = loss_fn(prediction, data.y)
            delta_G_loss += loss.item()
            mask = ~torch.isnan(data.f)
            valid_preds = des[mask]
            valid_labels = data.f[mask]
            loss_f = loss_fn(valid_preds, valid_labels)
            F_loss += loss_f.item()
            combined_features = torch.cat([des, batch_metal_features], dim=1)
            with torch.no_grad():
                combined_np = combined_features.cpu().numpy()
                t_pred = self.pi_model.predict(combined_np)  
                t_tensor = torch.tensor(t_pred, dtype=torch.float32, device=self.device)
                loss_rf = loss_fn(output, t_tensor.detach())
            pi_loss += loss_rf.item()

        if log_writer is not None:
            log_writer.add_scalar('DeltaG/' + loss_title, delta_G_loss / len(dataloader), epoch)
            log_writer.add_scalar('PI/' + loss_title, pi_loss / len(dataloader), epoch)
            log_writer.add_scalar('F/' + loss_title, F_loss / len(f_loader), epoch)

        return delta_G_loss / len(dataloader)

    def internal_pi_fit(self, num_epochs, tloader, vloader, test_loader=None, log_writer=None, pi_epoch=150, f_epoch=150, alpha=0.5):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        cur_epoch = -1
        min_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            for data, batch_metal_features in tloader:
                data = data.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)
                optimizer.zero_grad()

                output, des = self.model(data, batch_metal_features)
                prediction = output
                loss = loss_fn(prediction, data.y)
                if epoch >= pi_epoch:
                    combined_features = torch.cat([des, batch_metal_features], dim=1)
                    with torch.no_grad():
                        combined_np = combined_features.cpu().numpy()
                        t_pred = self.pi_model.predict(combined_np) 
                        t_tensor = torch.tensor(t_pred, dtype=torch.float32, device=self.device)
                    loss_rf = loss_fn(output, t_tensor.detach()) 

                    loss = loss + alpha * loss_rf
                loss.backward()
                optimizer.step()
            if epoch < f_epoch:
                for data in self.pi_loader:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data, None, True)
                    prediction = output
                    loss = loss_fn(prediction, data.y)
                    loss.backward()
                    optimizer.step()

            avg_train_loss = self.internal_pi_log(tloader, self.pi_loader, loss_fn, 'Train', log_writer, epoch)

            self.model.eval()
            avg_val_loss = self.internal_pi_log(vloader, self.pi_loader, loss_fn, 'Val', log_writer, epoch)
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                cur_epoch = epoch
            avg_test_loss = -1
            if test_loader is not None:
                avg_test_loss = self.internal_pi_log(test_loader, self.pi_loader, loss_fn, 'Test', log_writer, epoch)
            checkpoint_path = os.path.join(self.ckpt_path, f'model_epoch_{epoch + 1}.pt')
            torch.save(self.model.state_dict(), checkpoint_path)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        print(f"The best performing epoch is {cur_epoch + 1}.")
        print(f"Load epoch {cur_epoch + 1} model parameters")
        self.load_and_clean_checkpoints(cur_epoch)

    def internal_fit(self, num_epochs, tloader, vloader,test_loader=None, log_writer=None):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        cur_epoch = -1
        min_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for data, batch_metal_features in tloader:
                data = data.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)
                optimizer.zero_grad()

                output = self.model(data, batch_metal_features)
                if isinstance(output, tuple): 
                    prediction, logvar = output
                    loss = nll_loss(prediction, logvar, data.y)
                else:
                    prediction = output
                    loss = loss_fn(prediction, data.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(tloader)
            if log_writer is not None:
                log_writer.add_scalar('DeltaG/Train', avg_train_loss, epoch)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, batch_metal_features in vloader:
                    data = data.to(self.device)
                    batch_metal_features = batch_metal_features.to(self.device)
                    output = self.model(data, batch_metal_features)
                    if isinstance(output, tuple):
                        prediction, _ = output
                    else:
                        prediction = output
                    loss = loss_fn(prediction, data.y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(vloader)
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                cur_epoch = epoch
            if log_writer is not None:
                log_writer.add_scalar('DeltaG/Val', avg_val_loss, epoch)

            avg_test_loss = -1
            if test_loader is not None:
                test_loss = 0
                with torch.no_grad():
                    for data, batch_metal_features in test_loader:
                        data = data.to(self.device)
                        batch_metal_features = batch_metal_features.to(self.device)
                        output = self.model(data, batch_metal_features)
                        if isinstance(output, tuple):
                            prediction, _ = output
                        else:
                            prediction = output
                        loss = loss_fn(prediction, data.y)
                        test_loss += loss.item()
                avg_test_loss = test_loss / len(test_loader)
                if log_writer is not None:
                    log_writer.add_scalar('DeltaG/Test', avg_test_loss, epoch)

            checkpoint_path = os.path.join(self.ckpt_path, f'model_epoch_{epoch + 1}.pt')
            torch.save(self.model.state_dict(), checkpoint_path)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        print(f"The best performing epoch is {cur_epoch + 1}.")
        print(f"Load epoch {cur_epoch + 1} model parameters")
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

    def fit(self, num_epochs=100, pi_epoch=150, p=150, alpha=0.5):
        os.makedirs(self.log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_path)
        os.makedirs(self.ckpt_path, exist_ok=True)
        if self.pretrain_path is not None:
            self.model.load_pretrained_weights(self.pretrain_path, self.device)
        if self.PI:
            self.internal_pi_fit(num_epochs=num_epochs, tloader=self.train_loader, vloader=self.val_loader,
                              test_loader=self.test_loader, log_writer=writer, pi_epoch=pi_epoch, f_epoch=p, alpha=alpha)
        else:
            self.internal_fit(num_epochs=num_epochs, tloader=self.train_loader, vloader=self.val_loader,test_loader=self.test_loader, log_writer=writer)

    def cv_eval(self, num_epochs=100):
        os.makedirs(self.ckpt_path, exist_ok=True)
        cv_result = []
        for i in range(5):
            print(f"cv train fold {i + 1}")
            self.model = self.network(node_in_channels=self.in_channels, hidden_channels=self.gat_hidden_channels,
                                 out_channels=self.out_channels, num_node_layers=self.gat_layers,
                                 metal_feature_dim=self.metal_feature_dim,
                                 metal_embed_dim=self.metal_embed_dim, query_dim=self.query_dim,
                                 key_dim=self.key_dim, value_dim=self.value_dim,
                                 output_hidden_dim=self.output_hidden_dim).to(self.device)
            self.internal_fit(num_epochs=num_epochs, tloader=GATDeltaGModel.cv_train_loader[i], vloader=self.val_loader, log_writer=None)
            res = []
            with torch.no_grad():
                self.internal_predict(dataloader=GATDeltaGModel.cv_val_loader[i], results=res)
            res_df = pd.DataFrame(res)
            mse = mean_squared_error(res_df['predicted_value'], res_df['true_value'])
            print(f"cv fold {i + 1} mse: {mse}")
            cv_result.append(mse)
        return np.mean(cv_result)


    def internal_predict_epi(self, dataloader, results, n_samples=50):
        self.model.eval()
        self.model.regressor.train()
        for data, batch_metal_features in dataloader:
            data = data.to(self.device)
            batch_metal_features = batch_metal_features.to(self.device)
            preds_list = []
            logvar_list = []
            absolute_errors = []
            for _ in range(n_samples):
                output = self.model(data, batch_metal_features)
                if isinstance(output, tuple):
                    prediction, logvar = output
                    logvars = logvar.squeeze(-1).cpu().numpy()
                    logvar_list.append(logvars)
                else:
                    prediction = output
                predictions = prediction.squeeze(-1).cpu().numpy()
                current_absolute_errors = np.abs(predictions - data.y.cpu().numpy())
                preds_list.append(predictions)
                absolute_errors.append(current_absolute_errors)

            preds_list = np.array(preds_list).squeeze()
            absolute_errors = np.array(absolute_errors).squeeze()
            sum_absolute_errors = np.atleast_1d(absolute_errors.sum(axis=0))
            mean = np.atleast_1d(preds_list.mean(axis=0))
            uncertainty_epi = np.atleast_1d(preds_list.std(axis=0))

            labels = data.y.cpu().numpy()
            smiles_batch = data.smiles  
            if self.uncertainty:
                mean_logvar = np.atleast_1d(np.mean(logvar_list, axis=0))
                for smi, pred, label, epi, v, s in zip(smiles_batch, mean, labels, uncertainty_epi, mean_logvar, sum_absolute_errors):
                    results.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label,
                        'uncertainty_epi': epi,
                        'log_variance': v,
                        'sum_absolute_errors': s
                    })
            else:
                for smi, pred, label, epi in zip(smiles_batch, mean, labels, uncertainty_epi):
                    results.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label,
                        'uncertainty_epi': epi
                    })


    def internal_predict(self, dataloader, results, n_samples=50):
        self.model.eval()
        for data, batch_metal_features in dataloader:
            data = data.to(self.device)
            batch_metal_features = batch_metal_features.to(self.device)
            output = self.model(data, batch_metal_features)
            if isinstance(output, tuple):
                prediction, logvar = output
                logvars = logvar.squeeze(-1).cpu().numpy()
            else:
                prediction = output
            predictions = prediction.squeeze(-1).cpu().numpy()

            labels = data.y.cpu().numpy()

            smiles_batch = data.smiles 

            if self.uncertainty:
                for smi, pred, label, v in zip(smiles_batch, predictions, labels, logvars):
                    results.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label,
                        'log_variance': v
                    })
            else:
                for smi, pred, label in zip(smiles_batch, predictions, labels):
                    results.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label,
                    })

        result_uncertainty = {}
        self.model.regressor.train()
        for data, batch_metal_features in dataloader:
            data = data.to(self.device)
            batch_metal_features = batch_metal_features.to(self.device)
            preds_list = []
            logvar_list = []
            absolute_errors = []
            for _ in range(n_samples):
                output = self.model(data, batch_metal_features)
                if isinstance(output, tuple):
                    prediction, logvar = output
                    logvars = logvar.squeeze(-1).cpu().numpy()
                    logvar_list.append(logvars)
                else:
                    prediction = output
                predictions = prediction.squeeze(-1).cpu().numpy()
                current_absolute_errors = np.abs(predictions - data.y.cpu().numpy())
                preds_list.append(predictions)
                absolute_errors.append(current_absolute_errors)

            preds_list = np.array(preds_list).squeeze()
            uncertainty_epi = np.atleast_1d(preds_list.std(axis=0))

            smiles_batch = data.smiles
            for smi, epi in zip(smiles_batch, uncertainty_epi):
                result_uncertainty[smi] = epi
        for i in range(len(results)):
            results[i]['uncertainty'] = result_uncertainty[results[i]['smiles']]

    def internal_predict_PI(self, dataloader, results, n_samples=50):
        self.model.eval()
        for data, batch_metal_features in dataloader:
            data = data.to(self.device)
            batch_metal_features = batch_metal_features.to(self.device)
            output, _ = self.model(data, batch_metal_features)
            prediction = output
            predictions = prediction.squeeze(-1).cpu().numpy()

            labels = data.y.cpu().numpy()

            smiles_batch = data.smiles 

            if self.uncertainty:
                for smi, pred, label, v in zip(smiles_batch, predictions, labels, logvars):
                    results.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label,
                        'log_variance': v
                    })
            else:
                for smi, pred, label in zip(smiles_batch, predictions, labels):
                    results.append({
                        'smiles': smi,
                        'predicted_value': pred,
                        'true_value': label
                    })

        result_uncertainty = {}
        self.model.regressor.train()
        for data, batch_metal_features in dataloader:
            data = data.to(self.device)
            batch_metal_features = batch_metal_features.to(self.device)
            preds_list = []
            logvar_list = []
            absolute_errors = []
            for _ in range(n_samples):
                output = self.model(data, batch_metal_features)
                if isinstance(output, tuple):
                    prediction, logvar = output
                    logvars = logvar.squeeze(-1).cpu().numpy()
                    logvar_list.append(logvars)
                else:
                    prediction = output
                predictions = prediction.squeeze(-1).cpu().numpy()
                current_absolute_errors = np.abs(predictions - data.y.cpu().numpy())
                preds_list.append(predictions)
                absolute_errors.append(current_absolute_errors)

            preds_list = np.array(preds_list).squeeze()
            uncertainty_epi = np.atleast_1d(preds_list.std(axis=0))

            smiles_batch = data.smiles 
            for smi, epi in zip(smiles_batch, uncertainty_epi):
                result_uncertainty[smi] = epi
        for i in range(len(results)):
            results[i]['uncertainty'] = result_uncertainty[results[i]['smiles']]


    def predict_external_test(self, input_path : str, output_path : str, get_attention=False):


        external_test_loader = test_dataloader(input_path, 64)
        results = []
        with torch.no_grad():
            self.model.eval()
            for data, batch_metal_features in external_test_loader:
                data = data.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)
                if get_attention:
                    output, _, attention_weight = self.model(data, batch_metal_features, get_attention=True)
                else:
                    output, _ = self.model(data, batch_metal_features)
                prediction = output
                predictions = prediction.squeeze(-1).cpu().numpy()

                smiles_batch = data.smiles  
                compound_batch = data.compound
                metal_batch = data.metal
                if get_attention:
                    for idx, graph_weights in enumerate(attention_weight):
                        print(f"Graph {graph_weights['graph_idx']}:", sum(graph_weights["attention_weights"]))
                        mol = Chem.MolFromSmiles(smiles_batch[idx])

                        exclude_atoms = set()

                        pooh_smarts = "[P](=O)([O])"
                        pooh_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pooh_smarts))
                        if pooh_matches:
                            for match in pooh_matches:
                                if len(mol.GetAtomWithIdx(match[1]).GetNeighbors()) == 1 and len(
                                        mol.GetAtomWithIdx(match[2]).GetNeighbors()) == 1:
                                    exclude_atoms.add(match[0])
                                    exclude_atoms.add(match[1])
                                    exclude_atoms.add(match[2])

                        cooh_smarts = "[C](=O)([O])"
                        cooh_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(cooh_smarts))

                        if cooh_matches:
                            for match in cooh_matches:
                                if len(mol.GetAtomWithIdx(match[1]).GetNeighbors()) == 1 and len(
                                        mol.GetAtomWithIdx(match[2]).GetNeighbors()) == 1:
                                    exclude_atoms.add(match[0])
                                    exclude_atoms.add(match[1])
                                    exclude_atoms.add(match[2])

                        weights = []
                        for node_id, weight, atom in zip(graph_weights["node_ids"], graph_weights["attention_weights"],
                                                         mol.GetAtoms()):
                            # atom.SetProp('atomLabel', str(atom.GetSymbol()) + ":%.1f" % (weight * 100))
                            if atom.GetIdx() in exclude_atoms:
                                weights.append(0)
                            else:
                                weights.append(weight)
                            # weights.append(weight)
                            print(f"  Node {node_id}: Attention Weight = {weight}")
                        colors = weights_to_colormap(weights)
                        hit_ats = []
                        atom_cols = {}
                        for j, atom in enumerate(mol.GetAtoms()):
                            if atom.GetIdx() in exclude_atoms:
                                continue
                            hit_ats.append(atom.GetIdx())
                            atom_cols[atom.GetIdx()] = colors[j]
                        d = rdMolDraw2D.MolDraw2DSVG(500, 500)

                        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,
                                                           highlightAtomColors=atom_cols)
                        d.FinishDrawing()
                        svg = d.GetDrawingText()
                        with open(f'record/attention/example/{compound_batch[idx]}_{metal_batch[idx]}.svg', "w") as f:
                            f.write(svg)

                for smi, pred in zip(np.atleast_1d(smiles_batch), np.atleast_1d(predictions)):
                    results.append({
                        'smiles': smi,
                        'predicted_value': pred,
                    })
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)


    def get_feature_map(self, pi=True):
        results = []
        with torch.no_grad():
            self.model.eval()
            for data, batch_metal_features in self.test_loader:
                data = data.to(self.device)
                batch_metal_features = batch_metal_features.to(self.device)
                if pi:
                    output,_, hidden_output = self.model(data, batch_metal_features, export_hidden_feature=True)
                else:
                    output, hidden_output = self.model(data, batch_metal_features, export_hidden_feature=True)

                prediction = output
                predictions = prediction.squeeze(-1).cpu().numpy()

                hidden_features = hidden_output.cpu().numpy()  # [batch_size, 256]

                labels = data.y.cpu().numpy()
                metals = data.metal
                smiles_batch = data.smiles  

                batch_size = len(smiles_batch)
                for i in range(batch_size):
                    result_dict = {
                        'smiles': smiles_batch[i] if isinstance(smiles_batch, list) else np.atleast_1d(smiles_batch)[i],
                        'metal': metals[i],
                        'predicted_value': np.atleast_1d(predictions)[i],
                        'label': np.atleast_1d(labels)[i]
                    }

                    for j in range(hidden_features.shape[1]): 
                        feature_name = f'feature_{j}'
                        result_dict[feature_name] = hidden_features[i, j]

                    results.append(result_dict)

        results_df = pd.DataFrame(results)

        all_columns = list(results_df.columns)

        feature_cols = [col for col in all_columns if col.startswith('feature_')]
        non_feature_cols = [col for col in all_columns if col not in feature_cols]
        ordered_columns = non_feature_cols + sorted(feature_cols)
        results_df = results_df[ordered_columns]
        return results_df

    def get_attention(self, result_path):
        res_dic = {}
        self.model.eval()
        for data, batch_metal_features in self.train_loader:
            data = data.to(self.device)
            batch_metal_features = batch_metal_features.to(self.device)
            output, _, attention_weight = self.model(data, batch_metal_features, get_attention=True)
            smiles_batch = data.smiles
            compound_batch = data.compound
            metal_batch = data.metal

            for idx, graph_weights in enumerate(attention_weight):
                # print(f"Graph {graph_weights['graph_idx']}:")
                mol = Chem.MolFromSmiles(smiles_batch[idx])
                weights = []
                for node_id, weight, atom in zip(graph_weights["node_ids"], graph_weights["attention_weights"],
                                                 mol.GetAtoms()):
                    weights.append(str(weight))
                res_dic[f'{compound_batch[idx]}_{metal_batch[idx]}'] = ['|'.join(weights), smiles_batch[idx]]
        df = pd.DataFrame.from_dict(res_dic, orient='index')
        df = df.reset_index()
        columns_dict = {'index': 'original_compound', 0: 'attentions', 1: 'smiles'}
        df = df.rename(columns=columns_dict)
        df[['compound', 'metal', 'conformer']] = df['original_compound'].str.extract(
            r'^(CID_\d+)_deltaG_([A-Za-z]+)_(\d+)$')
        df.to_csv(result_path, index=False, encoding='utf-8-sig')



    def predict_train_test(self, result_train_path=None, result_val_path=None, result_test_path=None, epi=False):
        results_test = []
        results_val = []
        results_train = []
        with torch.no_grad():
            if self.PI:
                self.internal_predict_PI(self.test_loader, results_test)
                self.internal_predict_PI(self.val_loader, results_val)
                self.internal_predict_PI(self.train_loader, results_train)
            elif epi:
                self.internal_predict_epi(self.test_loader, results_test)
                self.internal_predict_epi(self.val_loader, results_val)
                self.internal_predict_epi(self.train_loader, results_train)
            else:
                self.internal_predict(self.test_loader, results_test)
                self.internal_predict(self.val_loader, results_val)
                self.internal_predict(self.train_loader, results_train)
        results_test_df = pd.DataFrame(results_test)
        results_val_df = pd.DataFrame(results_val)
        results_train_df = pd.DataFrame(results_train)
        if result_test_path is not None:
            results_test_df.to_csv(result_test_path, index=False)
        if result_train_path is not None:
            results_train_df.to_csv(result_train_path, index=False)
        if result_val_path is not None:
            results_val_df.to_csv(result_val_path, index=False)
        print("Train R2:", r2_score(results_train_df['true_value'], results_train_df['predicted_value']))
        print("Val R2:", r2_score(results_val_df['true_value'], results_val_df['predicted_value']))
        print("Test R2:", r2_score(results_test_df['true_value'], results_test_df['predicted_value']))
        print("Train mse:", mean_squared_error(results_train_df['true_value'], results_train_df['predicted_value']))
        print("Val mse:", mean_squared_error(results_val_df['true_value'], results_val_df['predicted_value']))
        print("Test mse:", mean_squared_error(results_test_df['true_value'], results_test_df['predicted_value']))
        print("Train uncertainty:", results_train_df['uncertainty'].mean(axis=0))
        print("Val uncertainty:", results_val_df['uncertainty'].mean(axis=0))
        print("Test uncertainty:", results_test_df['uncertainty'].mean(axis=0))
        return ((r2_score(results_train_df['true_value'], results_train_df['predicted_value']),
                r2_score(results_val_df['true_value'], results_val_df['predicted_value']),
                r2_score(results_test_df['true_value'], results_test_df['predicted_value'])),
                (mean_squared_error(results_train_df['true_value'], results_train_df['predicted_value']),
                 mean_squared_error(results_val_df['true_value'], results_val_df['predicted_value']),
                 mean_squared_error(results_test_df['true_value'], results_test_df['predicted_value'])),
                (results_train_df['uncertainty'].mean(axis=0),results_val_df['uncertainty'].mean(axis=0),results_test_df['uncertainty'].mean(axis=0)))

