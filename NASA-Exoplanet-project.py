import tkinter as tk
import torch as tr
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR
from tkinter import font
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import threading
from sklearn.metrics import classification_report, accuracy_score
import git
import os
project_path = os.path.expanduser("c:/Nasaproject")
os.makedirs(os.path.expanduser(project_path), exist_ok=True)
git.Git(project_path).clone("https://github.com/Akshobha/NASA-Spaceapps-challenge")


Project = tk.Tk()
Project.title("Exoplanet classifier")
Project.geometry("700x800")
Project.configure(bg='lightblue')
large_font = font.Font(family="Satoshi", size=24, weight="bold")

lable1 = tk.Label(Project, text="Guidelines", anchor="center")
lable1.pack()
lable2 = tk.Label(Project, text="1.Make sure your input file's in a csv format.", anchor="w",bg="light pink")
lable2.pack()
lable3 = tk.Label(Project, text="2.There should be no missing values in your input file.",bg="light pink")
lable3.pack()
lable4 = tk.Label(Project, text="3.Please read a README file in this path",bg="light pink")
lable4.pack()
lable5 = tk.Label(Project, text="4.Your device should have python, Scikit-learn,Numpy, Pytorch, Pandas and Gitpython installed.",bg="light pink")
lable5.pack()
lable6 = tk.Label(Project, text="Are you gonna input a dataset for a model to train on?",bg="light pink")
lable6.pack()
entry = tk.Entry(Project)
entry.pack()

def user_input1():
    user_input1 = entry.get()
    print(f"user_input1 received: {user_input1}")
    if user_input1.lower() == "yes":
        lable7 = tk.Label(Project, text="Input path of file for training(fill this field first and click submit):",bg="light pink")
        lable7.pack()
        entry2 = tk.Entry(Project)
        entry2.pack()

        def user_file():
            path = entry2.get().strip().strip('"').strip("'")
            print(f"File path received: {path}")
            global df
            df = pd.read_csv(path)
            print(f"Dataframe loaded with shape: {df.shape}")

        btn = tk.Button(Project, text="Submit", command=user_file)
        btn.pack()

        try:
            lable8 = tk.Label(Project, text="Can you input what's your target parameter(Fill this field second and click submit)",bg="light pink")
            lable8.pack()
            entry3 = tk.Entry(Project)
            entry3.pack()

            def user_target():
                target_columen = entry3.get().strip()
                print(f"Target column received: {target_columen}")
                global target_col
                target_col = [target_columen]
            btn2 = tk.Button(Project, text="Submit", command=user_target)
            btn2.pack()    

            lable9 = tk.Label(Project, text="Input your features(fill this field third and click submit)",bg="light pink")
            lable9.pack()
            entry4 = tk.Entry(Project)
            entry4.pack()
            
            def user_features():
                global feature_cols
                feature_columne = entry4.get()
                print(f"Feature columns received: {feature_columne}")
                global feature_cols
                feature_cols = [col.strip().strip('"').strip("'") for col in feature_columne.split(',')]
                df_dropped = df.dropna(subset=feature_cols + target_col)
                print(f"Dataframe after dropping NA has shape: {df_dropped.shape}")
                global le
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
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                class FNNClassifier(nn.Module):
                    def __init__(self, input_dim, output_dim):
                        super(FNNClassifier, self).__init__()
                        self.fc1 = nn.Linear(input_dim, 375)
                        self.act1 = nn.ReLU()
                        self.fc2 = nn.Linear(375, 250)
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

                input_dim = len(feature_cols)
                output_dim = len(np.unique(Y))
                model = FNNClassifier(input_dim, output_dim)
                global device
                device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
                from collections import Counter
                class_counts = Counter(y_train)
                num_classes = len(np.unique(y_train))
                counts = np.array([class_counts[i] for i in range(num_classes)])
                class_weights = 1.0 / counts
                class_weights = class_weights / class_weights.sum()
                class_weights_tensor = tr.tensor(class_weights, dtype=tr.float32)
                model = model.to(device)
                class_weights_tensor = class_weights_tensor.to(device)
                model.load_state_dict(tr.load(r"C:\Nasaproject\NASA-Spaceapps-challenge\parameter-values-for-model"))
                model.train()
                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                optimizer = tr.optim.SGD(model.parameters(), lr=1e-1)
                scheduler = ExponentialLR(optimizer,gamma=0.98)

                def update_label(report, accuracy):
                    def update():
                        lable10 = tk.Label(Project, text=f"Training completed.\nAccuracy: {accuracy:.4f}\n\n{report}")
                        lable10.pack()
                    Project.after(0, update)

                X_test_tensor = tr.tensor(X_test, dtype=tr.float32).to(device)
                y_test_tensor = tr.tensor(y_test, dtype=tr.long).to(device)

                def train_model():
                    epochs = 200
                    epoch_losses = []
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
                            total_loss += loss.item()
                        scheduler.step()
                        epoch_losses.append(total_loss)
                        if (epoch + 1) % 5 == 0:
                            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")
                    tr.save(model.state_dict(), r"C:\Nasaproject\NASA-Spaceapps-challenge\parameter-values-for-model")
                    model.eval()
                    with tr.no_grad():
                        outputs = model(X_test_tensor)
                        _, predicted = tr.max(outputs, 1)
                        y_pred = predicted.cpu().numpy()
                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, target_names=le.classes_)
                        update_label(report, accuracy)

                threading.Thread(target=train_model).start()
            btn3 = tk.Button(Project, text="Submit", command=user_features)
            btn3.pack()
            label12 = tk.Label(Project, text="Input a file to be classified(Fill this field last and then click Submit):",bg="light pink")
            label12.pack()
            entry5=tk.Entry(Project)
            entry5.pack()
            def user_classified():
                global device
                global le
                global feature_cols
                global user_classified_path
                user_classified_path = entry5.get().strip().strip('"').strip("'")
                cf = pd.read_csv(user_classified_path)
                cf = cf[feature_cols].copy()
                scaler = StandardScaler()
                cf_scaled = scaler.fit_transform(cf)
                X_user_tensor = tr.tensor(cf_scaled, dtype=tr.float32).to(device)

                class FNNClassifier(nn.Module):
                    def __init__(self, input_dim, output_dim):
                        super(FNNClassifier, self).__init__()
                        self.fc1 = nn.Linear(input_dim, 375)
                        self.act1 = nn.ReLU()
                        self.fc2 = nn.Linear(375, 250)
                        self.act2 = nn.ReLU()
                        self.fc3 = nn.Linear(250, 200)
                        self.act3 = nn.ReLU()
                        self.dropout = nn.Dropout(0.2)
                        self.fc4 = nn.Linear(200, 125)
                        self.act4 = nn.ReLU()
                        self.dropout =nn.Dropout(0.2)
                        self.output = nn.Linear(125, output_dim)

                    def forward(self, x):
                        x = self.dropout(self.act1(self.fc1(x)))
                        x = self.dropout(self.act2(self.fc2(x)))
                        x = self.dropout(self.act3(self.fc3(x)))
                        x = self.dropout(self.act4(self.fc4(x)))
                        x = self.output(x)
                        return x

                input_dim = len(feature_cols)
                output_dim = len(le.classes_)
                model = FNNClassifier(input_dim, output_dim).to(device)
                model.load_state_dict(tr.load(r"C:\Nasaproject\NASA-Spaceapps-challenge\parameter-values-for-model"))
                model.eval()

                with tr.no_grad():
                    outputs = model(X_user_tensor)
                    _, predicted = tr.max(outputs, 1)
                    y_pred = predicted.cpu().numpy()
                    predicted_classes = le.inverse_transform(y_pred)
                    result_text = f"Predictions:\n{predicted_classes}"

                cf["Classifications"] = pd.Series(predicted_classes)
                path = r"C:\Nasaproject\NASA-Spaceapps-challenge\classifications.csv"
                cf.to_csv(path, index=False)
            btn4 = tk.Button(Project, text="Submit", command=user_classified)
            btn4.pack()

        except Exception as e:
            tk.Label(Project, text=f"Error loading file: {e}", fg="red").pack()

    elif user_input1.lower() == "no":
        lable12 = tk.Label(Project, text="Input a dataset to classify",bg="light pink")
        lable12.pack()
        entry6 = tk.Entry(Project)
        entry6.pack()
        feature_cols=["pl_trandurh" ,"pl_rade","st_dist" , "st_tmag", "pl_orbper",
    "st_pmraerr1", "st_pmraerr2", "st_pmdec", "st_pmdecerr1", "st_pmdecerr2",
    "st_pmdeclim", "pl_tranmid", "pl_tranmiderr1", "pl_tranmiderr2", "pl_tranmidlim",
    "pl_orbpererr1", "pl_orbpererr2", "pl_trandurherr1", "pl_trandurherr2",
    "pl_radeerr1", "pl_radeerr2", "st_disterr1", "st_disterr2",
    "st_teff", "st_tefferr1"]
        device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
        le = LabelEncoder()

        def user_class():
            inp = entry6.get().strip().strip('"').strip("'")
            inp = inp.replace("\\", "\\\\")
            pf = pd.read_csv(inp)
            pf = pf[feature_cols].copy()
            scaler = StandardScaler()
            pf_scaled = scaler.fit_transform(pf)
            X_input_tensor = tr.tensor(pf_scaled, dtype=tr.float32).to(device)

            class FNNClassifier(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super(FNNClassifier, self).__init__()
                    self.fc1 = nn.Linear(input_dim, 375)
                    self.act1 = nn.ReLU()
                    self.fc2 = nn.Linear(375, 250)
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

            input_dim = len(feature_cols)
            output_dim = 6
            model = FNNClassifier(input_dim, output_dim).to(device)
            model.load_state_dict(tr.load(r"C:\Nasaproject\NASA-Spaceapps-challenge\parameter-values-for-model"))
            model.eval()

            with tr.no_grad():
                outputs = model(X_input_tensor)
                _, predicted = tr.max(outputs, 1)
                y_pred = predicted.cpu().numpy()
                class_names = ["APC","CP","FA","FP","KP","PC"]
                predicted_classes = [class_names[i] for i in y_pred]
                result_text = f"Predictions:\n{predicted_classes}"

            pf["Classifications"] = pd.Series(predicted_classes)
            path = r"C:\Nasaproject\NASA-Spaceapps-challenge\classifications.csv"
            pf.to_csv(path, index=False)

        btn5 = tk.Button(Project, text="Submit", command=user_class)
        btn5.pack()

    else:
        lable12 = tk.Label(Project, text="Please input a valid answer (yes/no)")
        lable12.pack()

btm = tk.Button(Project, text="Submit", command=user_input1)
btm.pack()
Project.mainloop()
