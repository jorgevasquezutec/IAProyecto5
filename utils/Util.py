import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from imblearn.metrics import sensitivity_score, specificity_score
import pandas as pd
import seaborn as sns


def show_example(img, label, dataset):
    print('Label: ',dataset.classes[label],"("+str(label)+")" )
    plt.imshow(img.permute(1,2,0))

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0).clamp(0,1))
        print(len(images))
        break

def denormalize (images,measn,std):
    means = torch.tensor(measn).reshape(1,3,1,1)
    stds = torch.tensor(std).reshape(1,3,1,1)
    return images*stds+means

def show_batch_denormalize(dl, mean, std):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denor_img = denormalize(images,mean,std)
        ax.imshow(make_grid(denor_img, nrow=8).permute(1, 2, 0).clamp(0,1))
        print(len(images))
        break

#Funciones de Ayuda para manejar si uno tiene cuda o CPU

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def plot_losses(history):
    train_losses = [x.get('loss_train') for x in history]
    val_losses = [x['error'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

def MatrixScore(y_test, y_pred,clases,model=''):
    matrix = confusion_matrix(y_test, y_pred)
    df2 = pd.DataFrame(matrix, index=clases, columns=clases)
    sns.heatmap(df2, annot=True, cbar=None, cmap="Greens")
    plt.title("Confusion Matrix"), plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("Real")

    if model != "":
        plt.savefig("img/"+model+"_"+""+".png")
    plt.show()

    print("Sensitivity")
    print(round(100*sensitivity_score(y_test, y_pred, average = 'micro'), 2))
    print("Specificity")
    print(round(100*specificity_score(y_test, y_pred, average = 'micro'), 2))
    print("F1-score")
    print(round(100*f1_score(y_test, y_pred, average = 'micro'), 2))
    print("Accuracy")
    print(round(100*accuracy_score(y_test, y_pred), 2))
    