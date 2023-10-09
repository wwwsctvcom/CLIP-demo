from pathlib import Path

class Arguments:

    def __init__(self):
        # for data
        self.dataset_path = Path.cwd() / "data"

        # for huggingface and pretrained model_name_or_path
        self.model_name_or_path = Path.cwd() / "model_name_or_path"

        # model_name_or_path saved
        self.model_trained = Path.cwd() / "model_trained"

        # for training
        self.resume = False
        self.resume_model_name_or_path = self.model_trained / "best"
        if self.resume and not self.resume_model_name_or_path.exists():
            self.resume_model_name_or_path.mkdir()

        self.resume_from_pth = False
        self.resume_from_pth_model = self.model_trained / "clip.pth"

        self.cuda = True
        self.accumulation_steps = 8
        self.distributed = False
        self.fp16 = False
        self.input_shape = [224, 224]
        self.pretrained = False
        self.batch_size = 64
        self.epochs = 5
        self.init_lr = 1e-2
        self.min_lr = self.init_lr * 0.01
        self.lr_decay_type = "cos"
        self.optimizer_type = "sgd"
        self.num_workers = 4

        self.optimizer_type = "sgd"
        self.momentum = 0.9
        self.weight_decay = 5e-4
