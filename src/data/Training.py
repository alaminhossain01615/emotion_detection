import torch

class Training:
    def __init__(self,model, optimizer, loss_func, device, save_path):
        self.model=model
        self.optimizer=optimizer
        self.loss_func=loss_func
        self.device=device
        self.save_path=save_path

        self.history={
            "train_loss":[],
            "train_acc":[],
            "test_loss":[],
            "test_acc":[]
        }

        self.best_test_loss = float("inf")

    def train_one_epoch(self,data):
        total_samples=0
        epoch_loss=0.0
        correct_pred=0
        self.model.train()
        for train_x,train_y in data:
            train_x, train_y = train_x.to(self.device), train_y.to(self.device)


            outputs=self.model(train_x)
            loss=self.loss_func(outputs,train_y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss=epoch_loss+(loss.item()*train_x.size(0))
            _, predicted=torch.max(outputs.data,1)
            total_samples=total_samples+train_x.size(0)
            correct_pred=correct_pred+(predicted==train_y).sum().item()

        epoch_loss=epoch_loss/total_samples
        epoch_acc=correct_pred/total_samples
        return epoch_loss, epoch_acc
    
    def validate_model(self,data):
        total_samples=0
        epoch_loss=0.0
        correct_pred=0
        self.model.eval()
        with torch.no_grad():
            for test_x,test_y in data:
                test_x,test_y=test_x.to(self.device), test_y.to(self.device)
                outputs=self.model(test_x)
                loss=self.loss_func(outputs, test_y)

                epoch_loss=epoch_loss+(loss.item())*test_x.size(0)
                _,predicted = torch.max(outputs.data,1)
                total_samples = total_samples+test_x.size(0)
                correct_pred=correct_pred+(predicted == test_y).sum().item()

        epoch_loss = epoch_loss/total_samples
        epoch_acc=correct_pred/total_samples
        return epoch_loss,epoch_acc
    
    def train_model(self,train_data,test_data,epochs):
        for epoch in range(1,50):
            print("---Training----")
            train_loss, train_acc = self.train_one_epoch(model,train_loader,loss_func,optimizer)
            test_loss, test_acc = self.validate_model(model,test_loader,loss_func)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(self.model.state_dict(),'model_weights.pth')

            print(f'Epoch {epoch} --- train_loss {train_loss}, train_acc {train_acc} --- test_loss {test_loss}, test_acc {test_acc}')
        return self.history