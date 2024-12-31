"""
Sets up wandb logging, tracking metrics, and model checkpointing
"""
import os
import sys
import numpy as np
import wandb
import torch
import torch.distributed as dist

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Parent_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Parent_DIR))

from metrics_utils import get_metric_function, AverageMeter
from model.model_utils import save_model

# -------------------------------------------------------------------------------------------------
class MetricManager(object):
    """
    Manages metrics and logging
    """
    
    def __init__(self, config):
        """
        @args:
            - config (Namespace): nested namespace containing all args
        """
        super().__init__()
        self.config = config
        self.device = config.device
        self.wandb_run = None

    def init_wandb(self):
        """
        Runs once at beginning of training if global_rank<=0 to initialize wandb object
        """
        self.wandb_run = wandb.init(project=self.config.project, 
                                    group=self.config.group,
                                    entity=self.config.wandb_entity, 
                                    config=self.config, 
                                    name=self.config.run_name, 
                                    notes=self.config.run_notes,
                                    dir=self.config.wandb_dir)

    def setup_wandb_and_metrics(self, rank):
        """
        Runs once at beginning of training for all processes to setup metrics 
        """

        # Set up common metrics depending on the task type
        task_type = self.config.task_type
        task_out_channel = self.config.no_out_channel

        if task_type =='class': 
            # Set up metric dicts, which we'll use during training to track metrics
            self.train_metrics = {'loss': AverageMeter(),
                                'auroc': AverageMeter()}
            self.eval_metrics = {'loss': AverageMeter(),
                                'acc_1': AverageMeter(),
                                'auroc': AverageMeter(),
                                'f1': AverageMeter()}
            
            # Define vars used by the metric functions
            if task_out_channel==1 or task_out_channel==2: # Assumes no multilabel problems
                self.metric_task = 'binary' 
            else: 
                self.metric_task = 'multiclass'
            self.multidim_average = 'global'
            data_range = None

            # Set up dictionary of functions mapped to each metric name
            self.train_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, self.metric_task, self.multidim_average, data_range).to(device=self.device) for metric_name in self.train_metrics if metric_name!='loss'}
            self.eval_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, self.metric_task, self.multidim_average, data_range).to(device=self.device) for metric_name in self.eval_metrics if metric_name!='loss'}

        elif task_type=='seg': 
            # Set up metric dicts, which we'll use during training to track metrics
            self.train_metrics = {'loss': AverageMeter(),
                                'f1': AverageMeter()}
            self.eval_metrics = {'loss': AverageMeter(),
                                'f1': AverageMeter()}
            
            # Define vars used by the metric functions
            if task_out_channel==1 or task_out_channel==2: # Assumes no multilabel problems
                self.metric_task = 'binary' 
            else: 
                self.metric_task = 'multiclass'
            self.multidim_average = 'samplewise'
            data_range = None

            # Set up dictionary of functions mapped to each metric name
            self.train_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, self.metric_task, self.multidim_average, data_range).to(device=self.device) for metric_name in self.train_metrics if metric_name!='loss'}
            self.eval_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, self.metric_task, self.multidim_average, data_range).to(device=self.device) for metric_name in self.eval_metrics if metric_name!='loss'}
        
        elif task_type=='enhance': 
            # Set up metric dicts, which we'll use during training to track metrics
            self.train_metrics = {'loss': AverageMeter(),
                                'ssim': AverageMeter(),
                                'psnr': AverageMeter()}
            self.eval_metrics = {'loss': AverageMeter(),
                                'ssim': AverageMeter(),
                                'psnr': AverageMeter()}
            
            # Define vars used by the metric functions 
            self.metric_task = 'multiclass' # Keep as multiclass for enhance applications
            self.multidim_average = 'global' # Keep as global for enhance applications
            data_range = None

            # Set up dictionary of functions mapped to each metric name
            self.train_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, self.metric_task, self.multidim_average, data_range).to(device=self.device) for metric_name in self.train_metrics if metric_name!='loss'}
            self.eval_metric_functions = {metric_name: get_metric_function(metric_name, task_out_channel, self.metric_task, self.multidim_average, data_range).to(device=self.device) for metric_name in self.eval_metrics if metric_name!='loss'}

        else:
            # Set up metric dicts, which we'll use during training to track metrics
            self.train_metrics = {'loss': AverageMeter()}
            self.eval_metrics = {'loss': AverageMeter()}

            # Add task metrics to the overall metrics
            self.train_metric_functions = None
            self.eval_metric_functions = None
            self.metric_task = None
            self.multidim_average = None
                
        if rank<=0:

            if self.wandb_run is not None:
                # Initialize metrics to track in wandb      
                self.wandb_run.define_metric("epoch")    
                for metric_name in self.train_metrics.keys():
                    self.wandb_run.define_metric(f'train_{metric_name}', step_metric='epoch')
                for metric_name in self.eval_metrics.keys():
                    self.wandb_run.define_metric(f'val_{metric_name}', step_metric='epoch')
            
            # Initialize metrics to track for checkpointing best-performing model
            self.best_val_loss = np.inf

    def on_train_epoch_start(self):
        """
        Runs on the start of each training epoch
        """

        # Reset metric values in AverageMeter
        for metric_name in self.train_metrics.keys():
            self.train_metrics[metric_name].reset()

    def on_train_step_end(self, loss, output, labels, rank, curr_lr):
        """
        Runs on the end of each training step
        """
        task_type = self.config.task_type

        # Adjust outputs to correct format for computing metrics
        if task_type=='class':
            output = torch.nn.functional.softmax(output, dim=1)
            if self.metric_task=='binary': 
                output = output[:,-1]
        
        elif task_type=='seg':
            output = torch.argmax(output,1)
            output = output.reshape(output.shape[0],-1)
            labels = labels.reshape(labels.shape[0],-1)
            
        elif task_type=='enhance':
            if labels.shape[2]==1: # 2D
                output = output[:,:,0,:,:]
                labels = labels[:,:,0,:,:]

        # Update train metrics based on the predictions this step
        for metric_name in self.train_metrics.keys():
            if metric_name=='loss':
                self.train_metrics[metric_name].update(loss, n=output.shape[0])
            else:
                metric_value = self.train_metric_functions[metric_name](output, labels)
                if self.multidim_average=='samplewise':
                    metric_value = torch.mean(metric_value)
                self.train_metrics[metric_name].update(metric_value.item(), n=output.shape[0])

        if rank<=0: 
            if self.wandb_run is not None: self.wandb_run.log({"lr": curr_lr})
            
    def on_train_epoch_end(self, model, optim, sched, epoch, rank):
        """
        Runs at the end of each training epoch
        """

        # Aggregate the measurements taken over each step
        if self.config.ddp:
            
            average_metrics = {}

            for metric_name in self.train_metrics.keys():

                batch_vals = torch.tensor(self.train_metrics[metric_name].vals).to(device=self.device)
                batch_counts = torch.tensor(self.train_metrics[metric_name].counts).to(device=self.device)
                batch_products = batch_vals * batch_counts

                dist.all_reduce(batch_products, op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(batch_counts, op=torch.distributed.ReduceOp.SUM)

                total_products = sum(batch_products)
                total_counts = sum(batch_counts)
                average_metrics[metric_name] = total_products.item() / total_counts.item()

        else:

            average_metrics = {}

            for metric_name in self.train_metrics.keys():
                average_metrics[metric_name] = self.train_metrics[metric_name].avg


        if rank<=0: # main or master process

            # Log the metrics for this epoch to wandb
            if self.wandb_run is not None: 
                for metric_name, avg_metric_val in average_metrics.items():
                    self.wandb_run.log({"epoch": epoch, f"train/{metric_name}": avg_metric_val}, commit=False)

            # Checkpoint the most recent model
            save_model(self.config, model, save_filename='model_last_epoch', epoch=epoch, optim=optim, sched=sched)
            
            # Checkpoint every epoch, if desired
            if epoch % self.config.checkpoint_frequency==0:
                save_model(self.config, model, save_filename=f'model_epoch_{epoch}', epoch=epoch, optim=optim, sched=sched)

        # Save the average metrics for this epoch into self.average_train_metrics
        self.average_train_metrics = average_metrics

    def on_eval_epoch_start(self):
        """
        Runs at the start of each evaluation loop
        """
        self.all_preds = []
        self.all_labels = []
        for metric_name in self.eval_metrics:
            self.eval_metrics[metric_name].reset()

    def on_eval_step_end(self, loss, output, labels, ids, rank, save_samples, split):
        """
        Runs at the end of each evaluation step
        """
        task_type = self.config.task_type

        # Adjust outputs to correct format for computing metrics
        if task_type=='class':
            output = torch.nn.functional.softmax(output, dim=1)
            if self.metric_task=='binary': 
                output = output[:,-1]
        
        elif task_type=='seg':
            output = torch.nn.functional.softmax(output, dim=1)
            output = torch.argmax(output,1)
            og_shape = output.shape[1:]
            output = output.reshape(output.shape[0],-1)
            labels = labels.reshape(labels.shape[0],-1)

        elif task_type=='enhance':
            if labels.shape[2]==1: # 2D
                output = output[:,:,0,:,:]
                labels = labels[:,:,0,:,:]

        # If exact_metrics was specified in the config, we'll save all the predictions so that we are computing exactly correct metrics over the entire eval set
        # If exact_metrics was not specified, then we'll average the metric over each eval step. Sometimes this produces the same result (e.g., average of losses over steps = average of loss over epoch), sometimes it does not (e.g., for auroc)
        if self.config.exact_metrics:
            if self.config.task_type=='class':
                self.all_preds += [output]
                self.all_labels += [labels]

            else:
                raise NotImplementedError('Exact metric computation not implemented for anything but class task type; not needed for average Dice or average loss.')
            
        # Update each metric with the outputs from this step 
        for metric_name in self.eval_metrics.keys():
            if metric_name=='loss':
                self.eval_metrics[metric_name].update(loss, n=output.shape[0])
            else:
                if not self.config.exact_metrics:
                    metric_value = self.eval_metric_functions[metric_name](output, labels)
                    if self.multidim_average=='samplewise':
                        metric_value = torch.mean(metric_value)
                    self.eval_metrics[metric_name].update(metric_value.item(), n=output.shape[0])

        # Save outputs if desired
        if save_samples:
            save_path = os.path.join(self.config.log_dir,self.config.run_name,'saved_samples',split)
            os.makedirs(save_path, exist_ok=True)
            for b_output, b_id in zip(output, ids):
                b_output = b_output.detach().cpu().numpy().astype('float32')
                if task_type=='seg':
                    b_output = b_output.reshape(og_shape)
                b_save_path = os.path.join(save_path,b_id+'_output.npy')
                np.save(b_save_path,b_output)

    def on_eval_epoch_end(self, rank, epoch, model, optim, sched, split, final_eval):
        """
        Runs at the end of the evaluation loop
        """

        # Directly compute metrics from saved predictions if using exact metrics
        if self.config.exact_metrics:

            # If not using distributed training, just gather all labels and predictions over the batches and compute metrics
            if not self.config.ddp:

                # Concatenate all preds and labels from different batches
                self.all_preds = torch.concatenate(self.all_preds)
                self.all_labels = torch.concatenate(self.all_labels)

                # Compute metrics over all preds and labels and store in average_metrics
                average_metrics = {}
                for metric_name in self.eval_metrics.keys():
                    if metric_name!='loss':
                        metric_value = self.eval_metric_functions[metric_name](self.all_preds, self.all_labels).item()
                        if self.multidim_average=='samplewise':
                            metric_value = torch.mean(metric_value)
                        self.eval_metrics[metric_name].update(metric_value, n=self.all_preds.shape[0])
                        average_metrics[metric_name] = metric_value
                    else:
                        average_metrics['loss'] = self.eval_metrics['loss'].avg

            # if using distributed training, need to gather labels and preds from all batches and all nodes
            else:
                torch.distributed.barrier()

                # Concat all batches on one node
                self.all_preds = torch.concatenate(self.all_preds)
                self.all_labels = torch.concatenate(self.all_labels)
                
                batch_losses = torch.tensor(self.eval_metrics['loss'].vals).to(device=self.device)
                batch_counts = torch.tensor(self.eval_metrics['loss'].counts).to(device=self.device)
                batch_products = batch_losses * batch_counts

                # Gather all tensors from different nodes on root node (rank 0)
                if rank == 0:
                    size = int(os.environ["WORLD_SIZE"])
                    prediction_list = [torch.empty_like(self.all_preds) for _ in range(size)]
                    labels_list = [torch.empty_like(self.all_labels) for _ in range(size)]

                    group = dist.group.WORLD
                    dist.gather(self.all_preds, gather_list=prediction_list, dst=0, group=group)
                    dist.gather(self.all_labels, gather_list=labels_list, dst=0, group=group)
                    
                else:
                    group = dist.group.WORLD
                    dist.gather(self.all_preds, dst=0, group=group)
                    dist.gather(self.all_labels, dst=0, group=group)

                dist.all_reduce(batch_products, op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(batch_counts, op=torch.distributed.ReduceOp.SUM)
                
                total_products = sum(batch_products)
                total_counts = sum(batch_counts)

                torch.distributed.barrier()

                # Compute metrics over all preds and labels and store in average_metrics
                if rank <= 0:
                    # Concatenate all tensors on root node
                    self.all_preds = torch.concatenate(prediction_list)
                    self.all_labels = torch.concatenate(labels_list)

                    # Compute each eval metric
                    average_metrics = {}
                    for metric_name in self.eval_metrics.keys():
                        if metric_name!='loss':
                            metric_value = self.eval_metric_functions[metric_name](self.all_preds, self.all_labels).item()
                            if self.multidim_average=='samplewise':
                                metric_value = torch.mean(metric_value)
                            self.eval_metrics[metric_name].update(metric_value, n=self.all_preds.shape[0])
                            average_metrics[metric_name] = metric_value
                        else:
                            average_metrics['loss'] = total_products.item() / total_counts.item()

        # Otherwise compute metrics as the average of metrics over all batches/nodes
        else:
            # If doing distributed training, average perfomrance over all batches and nodes
            if self.config.ddp:
                average_metrics = {}
                for metric_name in self.eval_metrics.keys():

                    batch_vals = torch.tensor(self.eval_metrics[metric_name].vals).to(device=self.device)
                    batch_counts = torch.tensor(self.eval_metrics[metric_name].counts).to(device=self.device)
                    batch_products = batch_vals * batch_counts

                    dist.all_reduce(batch_products, op=torch.distributed.ReduceOp.SUM)
                    dist.all_reduce(batch_counts, op=torch.distributed.ReduceOp.SUM)

                    total_products = sum(batch_products)
                    total_counts = sum(batch_counts)
                    average_metrics[metric_name] = total_products.item() / total_counts.item()

            # If not doing distributed training, average performance over all batches
            else:
                average_metrics = {metric_name: self.eval_metrics[metric_name].avg for metric_name in self.eval_metrics.keys()}

        # Checkpoint best models during training
        if rank<=0: 
            
            if not final_eval:

                # Update losses and determine whether to checkpoint this model
                checkpoint_model = False
                if average_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = average_metrics['loss']
                    checkpoint_model = True

                # Save model and update best metrics
                if checkpoint_model:
                    save_model(self.config, model, save_filename='model_best_checkpoint', epoch=epoch, optim=optim, sched=sched)

                # Update wandb with eval metrics
                self.wandb_run.log({"epoch":epoch, "best_loss": self.best_val_loss}, commit=False)
                for metric_name, avg_metric_eval in average_metrics.items():
                    self.wandb_run.log({"epoch":epoch, f"{split}/{metric_name}": avg_metric_eval}, commit=False)

            # Save the average metrics for this epoch into self.average_eval_metrics
            self.average_eval_metrics = average_metrics
        
    def on_training_end(self, rank, epoch, model_manager, optim, sched, ran_training):
        """
        Runs once when training finishes
        """
        if rank<=0: # main or master process
            
            if ran_training:
                # Log the best loss and metrics from the run and save final model
                self.wandb_run.summary["best_val_loss"] = self.best_val_loss

                # Final saves of backbone and tasks - commenting out, redundant with last_epoch saves
                # model_manager.save_entire_model(save_filename='final_model', epoch=epoch, optim=optim, sched=sched)
                # if self.config.save_model_components: model_manager.save_model_components(save_filename='final_model')
            
            # Finish the wandb run
            self.wandb_run.finish() 
        

def tests():
    pass

    
if __name__=="__main__":
    tests()