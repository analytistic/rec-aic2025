from dataclasses import dataclass, field


@dataclass
class Top1Rec:
    user_id_list: list = field(default_factory=list)
    top1_item_list: list = field(default_factory=list)


    def __post_init__(self):
        assert len(self.user_id_list) == len(self.top1_item_list), "user_id_list and top1_item_list must have the same length"

    def savecsv(self, filepath: str):
        import pandas as pd
        df = pd.DataFrame({
            'user_id': self.user_id_list,
            'book_id': self.top1_item_list,
        })
        df.sort_values('user_id', inplace=True)
        df.to_csv(filepath, index=False)