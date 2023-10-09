import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from config import Arguments
from utils.datasets import FlickrDataset
from utils.utils import *
from transformers import CLIPModel, CLIPProcessor


class Trainer:

    def __init__(self,
                 args: Arguments = None,
                 train_dataset: FlickrDataset = None,
                 test_dataset: FlickrDataset = None):
        self.args = args

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device {self.device}")

        # load huggingface model_name_or_path parameters
        self.processor = CLIPProcessor.from_pretrained(self.args.model_name_or_path)
        # self.model = CLIPModel(CLIPConfig())
        if not self.args.resume:
            if self.args.model_name_or_path:
                logger.info(f"Loading model from: {self.args.model_name_or_path}")
                self.model = CLIPModel.from_pretrained(args.model_name_or_path)
        elif self.args.resume_model_name_or_path.exists():
            logger.info(f"Loading model from: {self.args.resume_model_name_or_path}")
            self.model = CLIPModel.from_pretrained(args.resume_model_name_or_path)
        else:
            raise FileNotFoundError(f"model name or path not exists")

        if self.args.resume_from_pth and self.args.resume_from_pth_model:
            if self.args.resume_from_pth_model.suffix == ".pth":
                logger.info(f"Loading pretrained model state dict from: {str(self.args.resume_from_pth_model)}")
                model_dict = self.model.state_dict()
                pretrained_dict = torch.load(self.args.resume_from_pth_model, map_location=self.device)
                load_key, no_load_key, temp_dict = [], [], {}
                for k, v in pretrained_dict.items():
                    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                        temp_dict[k] = v
                        load_key.append(k)
                    else:
                        no_load_key.append(k)
                model_dict.update(temp_dict)
                self.model.load_state_dict(model_dict)
            else:
                raise ValueError(f"Unsupported model type {self.args.resume_from_pth_model}")

        # distributed
        self.scaler = None
        if self.args.fp16:
            from torch.cuda.amp import GradScaler as GradScaler
            logger.info("FP16")
            self.scaler = GradScaler()

        self.model_train = self.model.train().to(self.device)
        if self.args.cuda:
            self.model_train = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            self.model_train.cuda()

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    def train(self):
        logger.info("Start training...")
        optimizer = torch.optim.Adam(self.model_train.parameters(),
                                     lr=self.args.init_lr,
                                     betas=(self.args.momentum, 0.999),
                                     weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs,
                                                               eta_min=self.args.min_lr)

        loss_max = float("inf")
        for epoch in range(self.args.epochs):
            step_train = len(self.train_dataset) // self.args.batch_size
            step_test = len(self.test_dataset) // self.args.batch_size

            if step_train == 0 or step_test == 0:
                raise ValueError(f"step_train: {step_train} or step_test: {step_test} is too small")

            train_dataloader = DataLoader(self.train_dataset,
                                          shuffle=True,
                                          batch_size=self.args.batch_size,
                                          num_workers=self.args.num_workers,
                                          pin_memory=True,
                                          drop_last=True,
                                          collate_fn=collate_fn
                                          )
            test_dataloader = DataLoader(self.test_dataset,
                                         shuffle=False,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_workers,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=collate_fn
                                         )

            # training for one epoch
            train_total_loss = 0
            train_pbar = tqdm(total=step_train, desc=f'Epoch {epoch + 1}/{self.args.epochs}', postfix=dict, mininterval=0.3)
            for iteration, batch in enumerate(train_dataloader):
                if iteration >= step_train:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if not self.args.fp16:
                    outputs = self.model_train(**batch, return_loss=True)
                    loss = outputs.loss
                    loss.backward()  # backward, calculate gradient
                    optimizer.step()  # update parameter
                    optimizer.zero_grad()  # zero the gradient
                    scheduler.step()
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model_train(**batch)
                        loss = outputs.loss
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                train_total_loss += loss.item()
                train_pbar.set_postfix(**{'train average loss': train_total_loss / (iteration + 1), 'train loss': loss.item(), 'lr': get_lr(optimizer)})
                train_pbar.update(1)

            # saving best model_name_or_path per epoch
            if train_total_loss < loss_max:
                self.model.save_pretrained(self.args.model_trained / "best")
                self.processor.save_pretrained(self.args.model_trained / "best")
                loss_max = train_total_loss

            # test
            test_total_loss = 0
            test_pbar = tqdm(total=step_test, desc=f'Epoch {epoch + 1}/{self.args.epochs}', postfix=dict, mininterval=0.3)
            for iteration, batch in enumerate(test_dataloader):
                if iteration >= step_test:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model_train(**batch, return_loss=True)
                loss = outputs.loss
                test_total_loss += loss.item()
                test_pbar.set_postfix(**{'test average loss': test_total_loss / (iteration + 1), 'test loss': loss.item()})
                test_pbar.update(1)



if __name__ == "__main__":
    # set args
    arguments = Arguments()

    # reference from https://zhuanlan.zhihu.com/p/458809368
    seed_everything(42)

    logger.info("Loading dataset from: " + str(arguments.dataset_path))
    train_dataset = FlickrDataset(arguments.dataset_path,
                                  data_type="train",
                                  processor_path=arguments.model_name_or_path)
    test_dataset = FlickrDataset(arguments.dataset_path,
                                 data_type="test",
                                 processor_path=arguments.model_name_or_path)

    trainer = Trainer(args=arguments,
                      train_dataset=train_dataset,
                      test_dataset=test_dataset)
    trainer.train()
