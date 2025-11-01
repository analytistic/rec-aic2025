from src import Trainer
from config import BaseConfig



def main():

    cfg = BaseConfig(path='config/', args=None)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
