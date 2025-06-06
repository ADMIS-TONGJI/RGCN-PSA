import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        # self.pre_graph = pre_graph
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):   #验证集
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                # data = data[..., :self.args.dim_in]
                # label = target[..., :self.args.dim_out]
                label = target
                output, soft_clusters, clusters_G = self.model(data)
                label = self.scaler.inverse_transform(label)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                if self.args.device == 'cpu':
                    loss = self.loss(output, label)
                else:
                    loss = self.loss(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss, soft_clusters, clusters_G

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # data = data[..., :self.args.dim_in]
            # print("inputdata:")
            # print(data)
            # label = target[..., :self.args.dim_out]
            label = target
            self.optimizer.zero_grad()

            #NEW data and target shape: B, N, T; output shape: B, N, T
            output, soft_clusters, clusters_G = self.model(data)
            label = self.scaler.inverse_transform(label)
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)
            if self.args.device == 'cpu':
                loss = self.loss(output, label)
            else:
                loss = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm) #梯度裁剪
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss, soft_clusters, clusters_G

    def train(self):
        best_model = None
        best_cluster_labels = None
        best_loss = float('inf')
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss, train_soft_clusters, train_clusters_G = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss, val_soft_clusters, val_clusters_G = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            # if self.val_loader == None:
            #     val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
                        
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                # best_cluster_labels =  val_cluster_labels

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            # np.savez(self.best_cluster_labels_path, adj = best_cluster_labels.cpu().detach().numpy())

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),   #存放训练过程中需要学习的weight和bias
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        true_path = os.path.join(args.log_dir, '{}_true.npy'.format(args.dataset))
        pred_path = os.path.join(args.log_dir, '{}_pred.npy'.format(args.dataset))
        soft_clusters_path = os.path.join(args.log_dir, 'soft_clusters_{}.npz'.format(args.dataset))
        clusters_adj_path = os.path.join(args.log_dir, 'clusters_adj_{}.npz'.format(args.dataset))
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)   #将预训练的模型参数加载到新模型中
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                # data = data[..., :args.dim_in]
                # label = target[..., :args.dim_out]
                label = target
                output, soft_clusters, clusters_G = model(data)
                y_true.append(label)
                y_pred.append(output)
                soft_clusters_test = soft_clusters.cpu()
                clusters_G_test = clusters_G.cpu()
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        else:
            y_pred = torch.cat(y_pred, dim=0)
        np.save(true_path, y_true.cpu().numpy())
        np.save(pred_path, y_pred.cpu().numpy())
        np.savez(soft_clusters_path, adj = soft_clusters_test.detach().numpy())
        np.savez(clusters_adj_path, adj = clusters_G_test.detach().numpy())
        for t in range(y_true.shape[2]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, :, t], y_true[:, :, t], None, 0)
            logger.info("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, None, 0)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(mae, rmse, mape*100))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))