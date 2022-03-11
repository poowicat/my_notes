#### 错误

fatal: remote origin already exists

#### 解决

最后找到解决办法如下：

##### 1、先删除远程 Git 仓库

$ git remote rm origin

##### 2、再添加远程 Git 仓库

git remote add origin https://e.coding.net/poowicat/Storage/vdoing-blog.git

##### 3、推送本地仓库到远程

git push -u origin master

如果执行 git remote rm origin 报错的话，我们可以手动修改gitconfig文件的内容

$ vi .git/config
把 [remote “origin”] 那一行删掉就好了。

