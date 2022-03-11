### 问题：

本地推送到远程仓库，发现github改版本，由原来的默认master分支改变为main分支；故推送project到远程仓库时，出现如下错误：“error: src refspec main does not match any”

### 解决：

#### 1、分支不存在

使用checkout创建对应分支：

`git checkout -b master/main`

### 2、没有使用git add 和 git commit将文件添加至缓存

`git add .`

`git commit -m "xxx"`



### 3、没有README文件

`touch README`

**注意：**添加完 README文件后还需要重新git add 和 gitcommit

