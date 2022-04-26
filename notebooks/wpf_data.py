import pandas as pd


class WPFData:
    def __init__(
            self,
            dataframe: pd.DataFrame,
            input_len: int = 144,
            day_len: int = 144,
            train_days: int = 153,
            val_days: int = 16,
            test_days: int = 15,
            total_days: int = 184,
    ):
        super().__init__()
        self.dataframe = dataframe
        self.input_len = input_len
        self.total_size: int = total_days * day_len
        self.train_size: int = train_days * day_len
        self.val_size: int = val_days * day_len
        self.test_size: int = test_days * day_len
        self.type_map = {'all': 0, 'train': 1, 'val': 2, 'test': 3}

    def get(self, turbine_id: int, stage: str):
        # [border1s[0]: border2s[0]]: train_data (153*24*6, feature)
        # [border1s[1]: border2s[1]]: val_data (16*24*6, feature)
        # [border1s[2]: border2s[2]]: test_data (15*24*6, feature)
        border1s = [turbine_id * self.total_size,
                    turbine_id * self.total_size,
                    turbine_id * self.total_size + self.train_size - self.input_len,
                    turbine_id * self.total_size + self.train_size + self.val_size - self.input_len]
        border2s = [(turbine_id + 1) * self.total_size,
                    turbine_id * self.total_size + self.train_size,
                    turbine_id * self.total_size + self.train_size + self.val_size,
                    turbine_id * self.total_size + self.train_size + self.val_size + self.test_size]

        border1, border2 = border1s[self.type_map[stage]], border2s[self.type_map[stage]]
        return self.dataframe[border1:border2]
