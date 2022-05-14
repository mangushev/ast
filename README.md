# ast
Adversarial Sparse Transformer for Time Series Forecasting

paper : Adversarial Sparse Transformer for Time Series Forecasting


what has been done:

- Regular Transformer. Performance is just about 0.11 on gloss 50 metric on elect 1d. I arranged layer normalization after each layer calculation and dropouts are just before layer. Otherwise, it does not work for me. It seems, layer should see/have those dropout zeros without them being leveled by layer normalization. I'm getting similar performance with 2 or 3 layers, 4 heads and 128 width. It is likely that changing width and number of heads, could improve prformance. it is just taking in a range of 500 epochs for me. I notices that qloss training results high error for some items. I think rmse error wouldn't let this.
- Advesarial training is built, but it's not been used yet
- Sparse function is not implemented
- Prediction is done. Obviously, dataset should be continuation of the same time series! Probably, dataset should have TRAIN, VALIDATION, maybe TEST/PREDICT sets. 

Commands:

Prepare datasets:

python prepare_data.py --lookback_history=168 --estimate_length=24 2>&1 | tee logs/prepare_data-168-24-stride-20220427

Train:

python training.py --action=TRAIN --output_dir=checkpoints --lookback_history=168 --estimate_length=24 --train_epochs=1000 --learning_rate=1e-4 --minimal_rate=1e-5 --decay_steps=50000 --warmup_steps=50000 --clip_gradients=-1.0 --hidden_size=128 --feedforward_size=128 --embedding_size=20 --discriminator_lambda=0.0 --num_attention_heads=4 --num_hidden_layers=2 --dropout_prob=0.1 --num_series=370 --training_set_size=321598 --train_file=data/train.tfrecords --test_file=data/test.tfrecords --predict_file=data/test.tfrecords --train_scaler_file=data/train_scaler.joblib --test_scaler_file=data/test_scaler.joblib --predict_scaler_file=data/test_scaler.joblib --batch_size=64

Evaluate:

python training.py --action=EVALUATE --output_dir=checkpoints --lookback_history=168 --estimate_length=24 --train_epochs=1000 --learning_rate=1e-4 --minimal_rate=1e-5 --decay_steps=50000 --warmup_steps=50000 --clip_gradients=-1.0 --hidden_size=128 --feedforward_size=128 --embedding_size=20 --discriminator_lambda=0.0 --num_attention_heads=4 --num_hidden_layers=2 --dropout_prob=0.1 --num_series=370 --training_set_size=321598 --train_file=data/train.tfrecords --test_file=data/test.tfrecords --predict_file=data/test.tfrecords --train_scaler_file=data/train_scaler.joblib --test_scaler_file=data/test_scaler.joblib --predict_scaler_file=data/test_scaler.joblib --batch_size=64; cat output.csv

Predict:

python training.py --action=PREDICT --output_dir=checkpoints --lookback_history=168 --estimate_length=24 --train_epochs=1000 --learning_rate=1e-4 --minimal_rate=1e-5 --decay_steps=50000 --warmup_steps=50000 --clip_gradients=-1.0 --hidden_size=128 --feedforward_size=128 --embedding_size=20 --discriminator_lambda=0.0 --num_attention_heads=4 --num_hidden_layers=2 --dropout_prob=0.1 --num_series=370 --training_set_size=321598 --train_file=data/train.tfrecords --test_file=data/test.tfrecords --predict_file=data/test.tfrecords --train_scaler_file=data/train_scaler.joblib --test_scaler_file=data/test_scaler.joblib --predict_scaler_file=data/test_scaler.joblib --batch_size=64; less output.csv
