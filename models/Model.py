import torch
import torch.nn.functional as F
from zmq import device

class Model():
  def __init__(self, model):
    self.model = model
  def batch_results(self,outputs):
    batch_errors = [x['error'] for x in outputs]
    mean_error_batch = torch.stack(batch_errors).mean()
    batch_acc = [x['acc'] for x in outputs]
    mean_acc = torch.stack(batch_acc).mean()
    return {"error": mean_error_batch.item(),"acc": mean_acc.item()}

  def predict(self,img,classes,device,to_device):
    x = to_device(img.unsqueeze(0),device)
    output = self.model(x)
    _,pred = torch.max(output,dim=1)
    return classes[pred[0].item()]
   
  @torch.no_grad()
  def evaluate(self,val_loader,output=False):
    self.model.eval()
    outputsBatch = []
    outputArray = {
      'predicted': [],
      'labels': [],
    }
    for batch in val_loader:
      images,labels= batch
      outputs=self.model(images)
      loss = F.cross_entropy(outputs, labels)
      _, predicted = torch.max(outputs, dim=1)
      correct = torch.tensor(torch.sum(predicted == labels).item() / len(predicted))
      outputsBatch.append({"error": loss.detach(),"acc": correct })
      if(output):
        outputArray['predicted'].extend(predicted.detach().cpu().numpy())
        outputArray['labels'].extend(labels.detach().cpu().data.numpy())
    if(output):
        resOuput = self.batch_results(outputsBatch)
        resOuput['outputs'] = outputArray
        return resOuput
    return self.batch_results(outputsBatch)
    
  def train(self,fn_optimizer,loss_fn,num_epochs,train_loader,val_loader,learning_rate):
    log = []
    optimizer = fn_optimizer(self.model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
      self.model.train()
      train_losses = []
      for batch in train_loader:
        images,labels = batch
        output = self.model(images)
        loss = loss_fn(output,labels)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      result = self.evaluate(val_loader)
      result['loss_train'] = torch.stack(train_losses).mean().item()
      print('Epoch [{}/{}],Loss Train {:.4f},val_loss:{:.4f},Accuracy {:.4f}'.format(epoch+1, num_epochs,result['loss_train'],result['error'],result['acc']))
      log.append(result)
    return log


