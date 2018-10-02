python train.py HalfCheetah-v2 --mode PPO  --useScaler --iterSimBudget 16000 --nBenchmarkRuns 10
python train.py Reacher-v2 --mode PPO --useScaler --iterSimBudget 16000 --nBenchmarkRuns 10
python train.py Swimmer-v2 --mode PPO --useScaler --iterSimBudget 16000 --nBenchmarkRuns 10
python train.py InvertedPendulum-v2 --mode PPO --useScaler --iterSimBudget 16000 --nBenchmarkRuns 10