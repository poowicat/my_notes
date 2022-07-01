## Yolov5-v6.1网络与Yolov5-v5.0网络对比

- YOLOv5仓库是在`2020-05-18`创建的，到今天已经迭代了很多个大版本了；
- 现在（`2022-3-19`）已经迭代到`v6.1`了。

|              | **Yolov5 - v6.1**                                            |
| :----------: | :----------------------------------------------------------- |
|   **结构**   | 把网络的第一层（原来是`Focus`模块）换成了一个`6x6`大小的卷积层；<br/>将SPP改成SPPF后者效率更高（计算速度快了不止两倍）；<br/>最后的C3层引入shortcut和更新超参文件(作者在超参数文件中，将学习率0.2调为0.1；（3个超参文件，分别对应P5，NANO,P6三个版本模型）) |
|   **性能**   | Conv(k=6, s=2, p=2) 替换Focus，便于导出其他框架（for improved exportability）<br/>SPPF代替SPP，并且将SPPF放在主干最后一层（for reduced ops）<br/>主干中的C3层重复次数从9次减小到6次（for reduced ops）<br/>主干中最后一个C3层引入shortcut<br/> |
| **数据增强** | **Mosaic**<br/>**Copy paste**<br/>**Random affine**（随机进行仿射变换）<br/>**MixUp**<br/>**Albumentations**（主要是做些滤波、直方图均衡化以及改变图片质<br/>**Augment HSV**（随机调整色度，饱和度以及明度）；<br/>**Random horizontal flip**（随机水平翻转） |
| **推理对比** | ![image-20220406160651974](../../%E7%AC%94%E8%AE%B0/%E5%9B%BE%E7%89%87/image-20220406160651974.png)**总结：**<br/>在所有模型中，map 的性能从 + 0.3% 提高到 + 1.1% ，并且 ~5% 的浮点数减少可以略微提高速度和减少 cuda 内存<br/> |

**Refer：**

https://blog.csdn.net/weixin_44119362/article/details/120748319

https://github.com/ultralytics/yolov5/releases/tag/v6.1

