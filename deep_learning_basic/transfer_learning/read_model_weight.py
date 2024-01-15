import torch


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 3),
        )
        self.layer2 = torch.nn.Linear(3, 6)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(6, 7),
            torch.nn.Linear(7, 5),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


net = MyModel()
# print(net)

# -----------------------------------------------------------
# net.modules()、   net.named_modules()
# -----------------------------------------------------------
# for layer in net.modules():
#     # print(type(layer))
#     print(layer)
#     break


# for name, layer in net.named_modules():
#     print(name, type(layer))
    # print(name, layer)

# -----------------------------------------------------------
# net.children()、   net.named_children()
# -----------------------------------------------------------

# for layer in net.children():
#     print(layer)

# for name, layer in net.named_children():
#     print(name, layer)


# -----------------------------------------------------------
# net.parameters()、   net.named_parameters()
# -----------------------------------------------------------

# for param in net.parameters():
#     print(param.shape)


# for name, param in net.named_parameters():
#     print(name, param.shape)

# -----------------------------------------------------------
# net.state_dict()
# -----------------------------------------------------------

# for key, value in net.state_dict().items():
#     print(key, value.shape)



