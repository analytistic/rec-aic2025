from src import Trainer
from config import BaseConfig
from warnings import filterwarnings

filterwarnings('ignore')



def main():

    cfg = BaseConfig(path='config/', args={})
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.vail(topk=1)


if __name__ == "__main__":
    main()
