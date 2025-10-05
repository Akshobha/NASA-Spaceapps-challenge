import os
import git
import threading
import pandas as pd
import numpy as np
import torch as tr
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, render_template, request

app = Flask(__name__)

# === Clone repo at app startup (just like your Tkinter app) ===
project_path = os.path.expanduser("c:/Nasaproject")
if not os.path.exists(project_path):
    os.makedirs(project_path, exist_ok=True)
    print("Cloning NASA-Spaceapps-challenge repo...")
    git.Git(project_path).clone("https://github.com/Akshobha/NASA-Spaceapps-challenge")
else:
    print(f"Project path already exists at {project_path}, skipping clone.")

# Globals to hold data and state
df = None
target_col = None
feature_cols = None
le = None
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
model = None

# Helper class - your exact original FNNClassifier
class FNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 368)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(368, 250)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(250, 200)
        self.act3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(200, 125)
        self.act4 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(125, output_dim)

    def forward(self, x):
        x = self.dropout(self.act1(self.fc1(x)))
        x = self.dropout(self.act2(self.fc2(x)))
        x = self.dropout(self.act3(self.fc3(x)))
        x = self.dropout(self.act4(self.fc4(x)))
        x = self.output(x)
        return x

# Store results to show in UI
result_message = ""
training_done = False

@app.route("/", methods=["GET", "POST"])
def index():
    global df, target_col, feature_cols, le, model, device, result_message, training_done

    error = None
    step = 1  # UI step tracker for multi-stage input
    user_input1 = None

    if request.method == "POST":
        user_input1 = request.form.get("user_input1", "").strip().lower()

        if user_input1 == "yes":
            step = 2
            # Process training inputs and start training thread if submitted
            if "train_submit" in request.form:
                # Get training inputs from form
                train_file_path = request.form.get("train_file_path", "").strip().strip('"').strip("'")
                target_column = request.form.get("target_column", "").strip()
                features = request.form.get("features", "").strip()

                # Load CSV, prepare dataset
                try:
                    df = pd.read_csv(train_file_path)
                    target_col = [target_column]
                    feature_cols = [col.strip().strip('"').strip("'") for col in features.split(",")]

                    df_dropped = df.dropna(subset=feature_cols + target_col)
                    le = LabelEncoder()
                    df_dropped[target_col[0]] = le.fit_transform(df_dropped[target_col[0]])

                    X = df_dropped[feature_cols].values
                    Y = df_dropped[target_col[0]].values

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
                    )
                    X_train_tensor = tr.tensor(X_train, dtype=tr.float32)
                    y_train_tensor = tr.tensor(y_train, dtype=tr.long)
                    X_test_tensor = tr.tensor(X_test, dtype=tr.float32)
                    y_test_tensor = tr.tensor(y_test, dtype=tr.long)

                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

                    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

                    input_dim = len(feature_cols)
                    output_dim = len(np.unique(Y))

                    model = FNNClassifier(input_dim, output_dim).to(device)

                    from collections import Counter
                    class_counts = Counter(y_train)
                    num_classes = len(np.unique(y_train))
                    counts = np.array([class_counts[i] for i in range(num_classes)])
                    class_weights = 1.0 / counts
                    class_weights = class_weights / class_weights.sum()
                    class_weights_tensor = tr.tensor(class_weights, dtype=tr.float32).to(device)

                    model.load_state_dict(tr.load(os.path.join(project_path, "NASA-Spaceapps-challenge", "parameter-values-for-model")))
                    model.train()

                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                    optimizer = tr.optim.AdamW(model.parameters(), lr=1e-4)
                    scheduler = tr.optim.lr_scheduler.CyclicLR(
                        optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=40, mode="triangular", cycle_momentum=False
                    )

                    # Training function (runs in thread)
                    def train_model():
                        global result_message, training_done
                        epochs = 100
                        for epoch in range(epochs):
                            model.train()
                            total_loss = 0
                            for X_batch, y_batch in train_loader:
                                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                optimizer.zero_grad()
                                outputs = model(X_batch)
                                loss = criterion(outputs, y_batch)
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                                total_loss += loss.item()
                            if (epoch + 1) % 5 == 0:
                                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")
                        tr.save(model.state_dict(), os.path.join(project_path, "NASA-Spaceapps-challenge", "parameter-values-for-model"))
                        model.eval()
                        with tr.no_grad():
                            outputs = model(X_test_tensor.to(device))
                            _, predicted = tr.max(outputs, 1)
                            y_pred = predicted.cpu().numpy()
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, target_names=le.classes_)
                            result_message = f"Training completed.<br>Accuracy: {accuracy:.4f}<br><pre>{report}</pre>"
                        training_done = True

                    threading.Thread(target=train_model).start()

                    step = 3  # proceed to classification after training
                except Exception as e:
                    error = f"Error loading file or processing training data: {e}"

            elif "classify_submit" in request.form:
                # Classification after training
                classify_file_path = request.form.get("classify_file_path", "").strip().strip('"').strip("'")
                try:
                    cf = pd.read_csv(classify_file_path)
                    cf = cf[feature_cols].copy()
                    scaler = StandardScaler()
                    cf_scaled = scaler.fit_transform(cf)
                    X_user_tensor = tr.tensor(cf_scaled, dtype=tr.float32).to(device)

                    model.eval()
                    with tr.no_grad():
                        outputs = model(X_user_tensor)
                        _, predicted = tr.max(outputs, 1)
                        y_pred = predicted.cpu().numpy()
                        predicted_classes = le.inverse_transform(y_pred)
                        result_text = f"Predictions:\n{predicted_classes}"
                        cf["Classifications"] = pd.Series(predicted_classes)
                        out_path = os.path.join(project_path, "NASA-Spaceapps-challenge", "classifications.csv")
                        cf.to_csv(out_path, index=False)
                        result_message = f"Classification completed! Saved results to {out_path}."
                except Exception as e:
                    error = f"Error during classification: {e}"

        elif user_input1 == "no":
            step = 2
            feature_cols = [
                "pl_trandurh", "pl_rade", "st_dist", "st_tmag", "pl_orbper",
                "st_pmraerr1", "st_pmraerr2", "st_pmdec", "st_pmdecerr1", "st_pmdecerr2",
                "st_pmdeclim", "pl_tranmid", "pl_tranmiderr1", "pl_tranmiderr2", "pl_tranmidlim",
                "pl_orbpererr1", "pl_orbpererr2", "pl_trandurherr1", "pl_trandurherr2",
                "pl_radeerr1", "pl_radeerr2", "st_disterr1", "st_disterr2",
                "st_teff", "st_tefferr1"
            ]
            le = LabelEncoder()
            # Classification only
            if "classify_submit" in request.form:
                classify_file_path = request.form.get("classify_file_path", "").strip().strip('"').strip("'")
                try:
                    pf = pd.read_csv(classify_file_path)
                    pf = pf[feature_cols].copy()
                    scaler = StandardScaler()
                    pf_scaled = scaler.fit_transform(pf)
                    X_input_tensor = tr.tensor(pf_scaled, dtype=tr.float32).to(device)

                    input_dim = len(feature_cols)
                    output_dim = 6
                    model = FNNClassifier(input_dim, output_dim).to(device)
                    model.load_state_dict(tr.load(os.path.join(project_path, "NASA-Spaceapps-challenge", "parameter-values-for-model")))
                    model.eval()

                    with tr.no_grad():
                        outputs = model(X_input_tensor)
                        _, predicted = tr.max(outputs, 1)
                        y_pred = predicted.cpu().numpy()
                        class_names = ["APC", "CP", "FA", "FP", "KP", "PC"]
                        predicted_classes = [class_names[i] for i in y_pred]
                        result_text = f"Predictions:\n{predicted_classes}"
                        pf["Classifications"] = pd.Series(predicted_classes)
                        out_path = os.path.join(project_path, "NASA-Spaceapps-challenge", "classifications.csv")
                        pf.to_csv(out_path, index=False)
                        result_message = f"Classification completed! Saved results to {out_path}."
                except Exception as e:
                    error = f"Error during classification: {e}"

        else:
            # Anything else - show the exact error label like your Tkinter app
            error = "Please input a valid answer (yes/no)"

    return render_template("index.html",
                           error=error,
                           step=step,
                           result_message=result_message,
                           training_done=training_done)

if __name__ == "__main__":
    app.run(debug=True)
