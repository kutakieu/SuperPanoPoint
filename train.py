import hydra
from lightning import pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig

from superpanopoint.datasets import Dataset
from superpanopoint.lightning_wrapper import LightningWrapper


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    train_dataloader = Dataset().create_dataloader()
    val_dataloader = Dataset().create_dataloader()
    test_dataloader = Dataset().create_dataloader()

    model = LightningWrapper(cfg)

    early_stop_callback = EarlyStopping(
        monitor="loss_val",
        min_delta=0.01,
        patience=cfg.training.early_stopping_patience,
        verbose=False,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator.device,
        val_check_interval=1.0,
        devices=cfg.training.accelerator.gpus,
        log_every_n_steps=cfg.training.log_every_n_batch,
        max_epochs=cfg.training.n_epochs,
        callbacks=[early_stop_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
