# ML_2025
ML course project

# 目录结构
```plaintext
├── data                      # 数据集
|   ├── test_daily.csv        # 以天为单位处理后的测试集
|   ├── train_daily.csv       # 以天为单位处理后的训练集
│   └── origin_data.zip       # 未处理的测试集和训练集
├── results                   # 模型训练及预测结果
│   ├── lstm                  # LSTM模型结果
│   ├── transformer           # LSTM模型结果
│   └──  cnn_gru_transformer  # cnn_gru_transformer模型结果
├── main.py                   # 主程序
├── model.py                  # 三种模型的代码
├── utils.py                  # 工具函数
├── README.md
└── requirements.txt          # python环境
```
# 环境
- torch 2.4.1+cu118
- scikit-learn 1.3.2
- pandas 2.0.3
- numpy 1.24.3
- matplotlib 3.7.2
# 运行
```
python main.py
```
