import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from fcn_model import fcn_resnet50
from matplotlib import pyplot as plt


def main():
    aux = False  # inference time not need aux_classifier
    classes = 20

    # check files
    weights_path = './save_weights/model.pth'
    img_path = 'image.jpg'
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"weights {img_path} not found."


    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # preprocess image
    img_transforms = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])

    original_img = Image.open(img_path)
    img = img_transforms(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time.time()
        output = model(img.to(device))
        t_end = time.time()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # mask = Image.fromarray(prediction)
        # mask.save("predict_result.png")

    plt.subplot(121)
    plt.imshow(np.array(original_img))
    plt.subplot(122)
    plt.imshow(prediction)
    plt.show()


if __name__ == '__main__':
    main()
