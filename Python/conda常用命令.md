## **1. conda基本命令**

**检查Anaconda是否成功安装**

```
conda --version 
```

![img](../%E5%9B%BE%E7%89%87/1252882-20190402142706886-1863805247.png)

**检测目前安装了哪些环境**

```
conda info --envs 
```

![img](../%E5%9B%BE%E7%89%87/1252882-20190402142725673-576215713.png)

**检查目前有哪些版本的python可以安装：**

```
conda search --full-name python 
```

![img](../%E5%9B%BE%E7%89%87/1252882-20190402142953926-1705077319.png)

**安装不同版本的python**

```
对于GPU版本：conda create --name tensorflow-gpu python=3.6 

对于CPU版本：conda create --name tensorflow python=3.6 
```

**按照提示，激活**

```
activate tensorflow
```

**确保名叫tensorflow的环境已经被成功添加：**

```
conda info --envs
```

**检查新环境中的python版本**　

```
python --version 
```

**升级**

```
conda update conda
conda update anaconda
conda update anaconda-navigator    //update最新版本的anaconda-navigator  
```

**卸载**

```
计算机控制面板->程序与应用->卸载        //windows
rm -rf anaconda    //ubuntu
```

**最后，建议清理下.bashrc中的Anaconda路径。**


**conda环境使用基本命令：**

```
conda update -n base conda        //update最新版本的conda
conda create -n xxxx python=3.5   //创建python3.5的xxxx虚拟环境
conda activate xxxx               //开启xxxx环境
conda deactivate                  //关闭环境
conda env list                    //显示所有的虚拟环境
```

**anaconda安装最新的TensorFlow版本**

- 打开anaconda-prompt

- 查看tensorflow各个版本：（查看会发现有一大堆TensorFlow源，但是不能随便选，选择可以用查找命令定位）

  ```
  anaconda search -t conda tensorflow
  ```

- 找到自己安装环境对应的最新TensorFlow后（可以在终端搜索anaconda，定位到那一行），然后查看指定包

  ```
  anaconda show <USER/PACKAGE>
  ```

- 查看tensorflow版本信息

  ```
  anaconda show anaconda/tensorflow
  ```

- 第4步会提供一个下载地址，使用下面命令就可安装1.8.0版本tensorflow

  ```
  conda install --channel https://conda.anaconda.org/anaconda tensorflow=1.8.0 
  ```

**更新、卸载安装包**

```
conda list         #查看已经安装的文件包
conda update xxx   #更新xxx文件包
conda uninstall xxx   #卸载xxx文件包
```

**删除虚拟环境**

```
conda remove -n xxxx --all //创建xxxx虚拟环境
```

**清理（conda瘦身）**

`conda clean`就可以轻松搞定！

第一步：通过`conda clean -p`来删除一些没用的包，这个命令会检查哪些包没有在包缓存中被硬依赖到其他地方，并删除它们。

第二步：通过`conda clean -t`可以将conda保存下来的tar包。

```
conda clean -p      //删除没有用的包
conda clean -t      //tar打包
```

## **2. jupyter notebook默认工作目录设置**

参考：https://blog.csdn.net/liwei1205/article/details/78818568
1）在Anaconda Prompt终端中输入下面命令，查看你的notebook配置文件在哪里：

```
jupyter notebook --generate-config
//会生成文件C:Users用户.jupyterjupyter_notebook_config.py
```

2）打开jupyter_notebook_config.py文件通过搜索关键词：c.NotebookApp.notebook_dir，修改如下　　

```
c.NotebookApp.notebook_dir = 'E:\tf_models'     //修改到自定义文件夹
```

3）然后重启notebook服务器就可以了
注：其它方法直接命令到指定目录，Anaconda Prompt终端中输：jupyter notebook 目录地址　　

**参考文献：**

【1】[conda常用命令](https://www.jianshu.com/p/7ebe1df808ba)