from torch.utils.data import TensorDataset, DataLoader
import torch 
from..tools.tools import AverageMeter, accuracy_topk


class Trainer():
    '''
    All training functionality
    '''
    def __init__(self, device, model, optimizer, criterion, scheduler):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
    
    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
        '''
        Run one train epoch
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to train mode
        model.train()

        for i, (ids, mask, y) in enumerate(train_loader):

            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)
            # Forward pass
            logits = model(ids, mask)
            loss = criterion(logits, y)

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, y)
            accs.update(acc.item(), ids.size(0))
            losses.update(loss.item(), ids.size(0))

            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tAccuracy {accs.val:.3f} ({accs.avg:.3f})')


    @staticmethod
    def eval(val_loader, model, criterion, device, return_logits=False):
        '''
        Run evaluation
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to eval mode
        model.eval()

        all_logits = []
        with torch.no_grad():
            for (ids, mask, y) in val_loader:

                ids = ids.to(device)
                mask = mask.to(device)
                y = y.to(device)

                # Forward pass
                logits = model(ids, mask)
                all_logits.append(logits)
                loss = criterion(logits, y)

                # measure accuracy and record loss
                acc = accuracy_topk(logits.data, y)
                accs.update(acc.item(), ids.size(0))
                losses.update(loss.item(), ids.size(0))

        if return_logits:
            return torch.cat(all_logits, dim=0).detach().cpu()

        print(f'Test\t Loss ({losses.avg:.4f})\tAccuracy ({accs.avg:.3f})\n')
        return accs.avg

    @staticmethod
    def split_and_dl(model, sentences, y, val=0.2, bs=8):
        '''
            Prep data into a train and val dataloader
        '''
        # Get ids and mask
        inputs = model.tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt")
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        y = torch.LongTensor(y)

        # split
        num_val = int(val*ids.size(0))
        ids_train = ids[num_val:]
        mask_train = mask[num_val:]
        y_train = y[num_val:]
        ids_val = ids[:num_val]
        mask_val = mask[:num_val]
        y_val = y[:num_val]

        # Use dataloader to handle batches
        train_ds = TensorDataset(ids_train, mask_train, y_train)
        val_ds = TensorDataset(ids_val, mask_val, y_val)

        if len(ids_train)>0:
            train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        else: train_dl = None
        if len(ids_val)>0:
            val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
        else: val_dl = None
        return train_dl, val_dl
    
    def train_process(self, sentences, y, save_path, max_epochs=10, bs=8):

        train_dl, val_dl = self.split_and_dl(self.model, sentences, y, bs=bs)

        best_acc = 0
        for epoch in range(max_epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train(train_dl, self.model, self.criterion, self.optimizer, epoch, self.device)
            self.scheduler.step()

            # evaluate on validation set
            acc = self.eval(val_dl, self.model, self.criterion, self.device)
            if acc > best_acc:
                best_acc = acc
                state = self.model.state_dict()
                torch.save(state, save_path)
