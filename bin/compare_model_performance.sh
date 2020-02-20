source venv/bin/activate
cd sood
nohup python exp/compare_model_performance.py -m fb -d low> log/feature_bagging_batch_test_20200219.log &


