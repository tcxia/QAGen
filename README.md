# QAGen

## 训练模型
```python
python main.py
```

## 生成qa对
```python
python generate_qa.py
```

## QAEvalution
```python
python main.py --pretrain_file $PATH_FILE_QAEVAL --unlabel_ratio 0.1 --lazy_loader --batch_size 32
```

## Semi-learning for SQuAD
```python
python main.py --pretrian_file $PATH_FILE_semiEVAL --unlabel_ratio 0.1 --lazy_loader --batch_size 32
```

