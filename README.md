# Rescue the General 

A mixed competitive-cooperative multi-agent reinforcement learning environment for Gym.

To reproduce results of the ICML paper on a single GPU sequentially run

```
python ICML_paper_experiments.py
```

For faster results, execute each run in parallel on a multiple GPUs, setting --device appropriately. On 4 RTX2080s this should take 4-5 days if two jobs per card are run simultaneously. 
