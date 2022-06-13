import wandb

class Wandb:
    def __init__(self, project, flag, name, config):
        self.flag = flag
        if self.flag:
            wandb.init(project=project, name=name, entity="myeongu", config=config)
    
    def write_log(self, epoch, cur_loss, best_loss):
        if self.flag:
            wandb.log({"epoch": epoch, "current loss": cur_loss, "best loss":  best_loss})
            # wandb.log({"Train/loss": train_loss, "Train/accuracy": train_acc, "learning rate": current_lr})

    def watch(self, model):
        if self.flag:
            wandb.watch(model)