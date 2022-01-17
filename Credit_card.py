import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

#print(sys.version)
#print(torch.version)
#print('cuda:', torch.version.cuda)

# Choose cpu/gpu
use_gpu = 0
if (use_gpu):
    print('\nEnable gpu')
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")  # Uncomment this to run on GPU

else:
    print('\nRun on cpu')
    dtype = torch.FloatTensor
    device = torch.device("cpu")

df = pd.read_csv('creditcard.csv')

normal = df[df['Class']==0]
fraud = df[df['Class']==1]
normal = normal.drop('Class', axis=1)
fraud = fraud.drop('Class', axis=1)

normal_train, normal_test = train_test_split(normal,test_size=0.5,random_state=3)

scaler = MinMaxScaler()

normal_train_fit = scaler.fit_transform(normal_train)
normal_test_fit = scaler.transform(normal_test)
fraud_fit = scaler.transform(fraud)

normal_train_fit_train, normal_train_fit_val = train_test_split(normal_train_fit,test_size=0.2,random_state=3)

num_epochs = 200
batch_size = 2048
hidden_layer1 = 30
hidden_layer2 = 25
encoding_dim = 20

input_dim = normal_train_fit_train.shape[1]

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1),
            nn.Sigmoid(),
            nn.Linear(hidden_layer1, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, hidden_layer2),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_layer2, encoding_dim),
            nn.Sigmoid(),
            nn.Linear(encoding_dim, hidden_layer1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Settings
if (use_gpu):
    model = autoencoder().cuda() # enable GPU
else:
    model = autoencoder()

# For training on normal samples
train_loader = torch.utils.data.DataLoader(dataset=normal_train_fit_train,
                                          batch_size=batch_size,
                                          shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=normal_train_fit_val,
                                          batch_size=batch_size,
                                          shuffle=True)

# For testing on fraud examples (shuffle=False)
test_fraud_loader = torch.utils.data.DataLoader(dataset=fraud_fit,
                                          batch_size=batch_size,
                                          shuffle=False)

# For testing on unseen normal sample (shuffle=False)
test_normal_loader = torch.utils.data.DataLoader(dataset=normal_test_fit,
                                          batch_size=batch_size,
                                          shuffle=False)

criterion =  nn.MSELoss()

optimizer =  torch.optim.Adam(model.parameters(), lr=0.001)

loss_train = []
loss_val = []

for epoch in range(num_epochs):

    ###################
    # train the model #
    ###################
    model.train()  # prepare model for training
    for data in train_loader:
        data = data.type(dtype)
        output = model(data)
        loss = criterion(output, data)
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    ######################
    # validate the model #
    ######################
    model.eval()  # prepare model for evaluation
    for data in valid_loader:
        data = data.type(dtype)
        output = model(data)
        loss = criterion(output, data)
        loss_val.append(loss.item())
        loss.backward()

X1 = np.arange(0, len(loss_val),1)
X2 = np.arange(0, len(loss_train),1)
plt.plot(X1, loss_val, label ='Loss Validation set')
plt.plot(X2, loss_train, label ='Loss Training set')
plt.legend()
plt.show()

model.eval() # Sets the module in evaluation mode.
model.cpu()  # Moves all model parameters and buffers to the CPU to avoid out of memory

# Normal test dataset
test_normal_distance = []
for data in test_normal_loader:
    data = data.type(dtype).cpu().detach()
    output = model(data)
    test_normal_distance += torch.sqrt((torch.sum((data-output)**2,axis=1)))

test_normal_distance = torch.FloatTensor(test_normal_distance)
test_normal_distance = test_normal_distance.numpy()

# Fraud test dataset
test_fraud_distance = []
for data in test_fraud_loader:
    data = data.type(dtype).cpu().detach()
    output = model(data)
    test_fraud_distance += torch.sqrt((torch.sum((data-output)**2,axis=1)))

test_fraud_distance = torch.FloatTensor(test_fraud_distance)
test_fraud_distance = test_fraud_distance.numpy()

##################
#CONFUSION MATRIX#
##################
from sklearn.metrics import confusion_matrix

LABELS = ["Normal", "Fraud"]

target = np.concatenate((np.zeros(x_test_normal.shape[0]),np.ones(x_test_fraud.shape[0])))
scores = np.concatenate((test_normal_distance,test_fraud_distance))

threshold = 0.75

y_pred = [1 if e > threshold else 0 for e in scores]
conf_matrix = confusion_matrix(target, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

NN = conf_matrix[0,0]
NF = conf_matrix[0,1]
FN = conf_matrix[1,0]
FF = conf_matrix[1,1]
print('False positive rate = %.2f %%' % (FN/(FN+FF)*100))
print('True positive rate = %.2f %%' % (NN/(NN+NF)*100))

###########
#ROC CURVE#
###########

from sklearn.metrics import roc_curve, auc

target = np.concatenate((np.zeros(x_test_normal.shape[0]), np.ones(x_test_fraud.shape[0])))
scores = np.concatenate((test_normal_distance, test_fraud_distance))

plt.figure(figsize=(7, 7))

fp, vp, thresholds = roc_curve(target, scores, pos_label=1)
roc_auc = auc(fp, vp)

plt.plot(fp, vp, color='red', label='ROC curve %s (AUC = %0.4f)' % ('AE', roc_auc))

plt.xlabel('False Positive', fontsize=16)
plt.ylabel('True Positive', fontsize=16)
plt.plot([0, 1], [0, 1],
         linestyle='--', color=(0.6, 0.6, 0.6),
         label='Random guess')

plt.grid()
plt.legend(loc="best", fontsize=16)
plt.tight_layout()
#plt.savefig("images/ROC.png")