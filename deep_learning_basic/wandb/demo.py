import wandb
import random
import datetime

run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(
    project="wandb_demo",       # 项目名
    name=f"run-{run_name}",     # 本次 run 的名称
    config={                    # 记录需要跟踪的超参数
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    })

print("hi")
# 模拟训练过程，并记录相关指标 ： acc， loss
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # 记录相关指标
    wandb.log({"acc": acc, "loss": loss})

print("bye")
# 结束 wandb
wandb.finish()
