from src import Trainer, Infer
from config import BaseConfig
from warnings import filterwarnings

filterwarnings('ignore')



def main():

    cfg = BaseConfig(path='config/', args={})
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.vail(topk=1)

    infer = Infer(cfg)
    top1_rec = infer.predict(topk=1)
    top1_rec.savecsv('submission.csv')


if __name__ == "__main__":
    main()
