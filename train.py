import hydra
from lightning import pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig

from superpanopoint import Settings
from superpanopoint.datasets.dataset_factory import DatasetFactory
from superpanopoint.lightning_wrapper import LightningWrapper


@hydra.main(
    config_path=Settings().config_dir,
    config_name=Settings().config_name,
    version_base=None,
)
def main(cfg: DictConfig):
    dataset_factory = DatasetFactory(cfg)
    train_dataloader = dataset_factory.create_dataset("train").create_dataloader()
    val_dataloader = dataset_factory.create_dataset("val").create_dataloader()
    test_dataloader = dataset_factory.create_dataset("test").create_dataloader()

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
