1、数据集 \
   数据集使用 【花分类数据集】 \
   下载地址：[https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) \
   配置数据集地址 ：将`train.py`文件中 设置的默认数据集地址 `--data-path` 改成你存放数据集`flower_photos`文件夹的绝对路径

2、预训练权重   \
   在`vit_model.py`文件中, 每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重 \
   视频课程中使用的是 [vit_base_patch16_224](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth) \
   配置预训练权重地址 : 在`train.py`文件中，将`--weights`参数的默认地址设置成你存放 预训练权重的路径 
   

3、训练 \
   运行`train.py`脚本，开始训练后，会自动生成`class_indices.json`文件，其内容是 花的类别与索引 


4、预测 \
   在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下) \
   在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径 

