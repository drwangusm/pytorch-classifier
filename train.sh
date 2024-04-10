#1.分割数据集，比如我想验证集比例是0.1
python processing.py --val_size 0.1

#2.训练ghostnet
python main.py --model_name ghostnet --pretrained --config config/config.py --save_path runs/ghostnet_flower --lr 1e-4 --warmup --amp --imagenet_meanstd  --Augment AutoAugment 

#3.演示如何使用metrice.py进行检测
python metrice.py --task test --save_path runs/ghostnet_flower

#4.可视化数据集的识别情况和tsne可视化，也是只需要添加两个参数 --visual --tsne
python metrice.py --task test --save_path runs/ghostnet_flower --visual --tsne

#5.预测 
python predict.py --source dataset/test/00 --save_path runs/ghostnet_flower

#6.predict.py文件结合pytorch_grad_cam库实现了热力图可视化，并支持多种热力图计算方法
python predict.py --source dataset/test/00 --save_path runs/ghostnet_flower --cam_visual --cam_type GradCAMPlusPlus




