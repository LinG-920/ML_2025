
# 导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def preprocess(input_path, flag, output_path):
    column_names = [
            'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
            'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
        ]
    # 读入数据文件
    if flag :
        dataset = pd.read_csv(input_path, header=None, names=column_names,
                        parse_dates=[0], index_col=[0])
    else:
        dataset = pd.read_csv(input_path, header=0, parse_dates=[0], index_col=[0])

    print(dataset.shape)
    print(dataset.head(10))

    # 异常值处理，inplace=True，在原地替换，不创建新的df
    dataset.replace('?', np.nan, inplace=True)
    dataset = dataset.astype('float32')

    dataset.ffill(inplace=True)


    dataset['sub_metering_remainder'] = (dataset['Global_active_power'] * 1000 / 60) - \
                                        (dataset['Sub_metering_1'] + dataset['Sub_metering_2'] + dataset['Sub_metering_3'])

    print(dataset.shape)
    print(dataset.head(10))

    # 4. 按天聚合数据
        # 定义聚合规则
    agg_rules = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'sub_metering_remainder': 'sum',
        'RR': 'median',  # 天气数据在一天内是恒定的，取第一个值即可
        'NBJRR1': 'median',
        'NBJRR5': 'median',                       
        'NBJRR10': 'median',
        'NBJBROU': 'median'
    }


    daily_data = dataset.resample('D').agg(agg_rules)
    print(daily_data)
    daily_data.to_csv(output_path)

if __name__=="__main__":
    preprocess("data/train.csv", False, "data/train_daily.csv")

    preprocess("data/test.csv", True, "data/test_daily.csv")



