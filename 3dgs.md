# **3D Gaussian Splatting for Real-Time Radiance Field Rendering**

基于splatting和机器学习的三维重建方法

## 整体框架

![image-20241203103940733](assets/image-20241203103940733.png)

1. 通过**sfm**获得初始化稀疏点云（colmap）
2. 基于初始化点云生成**3D高斯椭球**集
3. 相机模型的投影矩阵
4. 可微分渲染
5. Loss比对迭代（点云位置，点云颜色，不透明度，高斯椭球的协方差矩阵）
6. 基于梯度自适应改变点云的分布方式

**整个3dgs的流程可以形象化地表现为3个模块：**

* 选择雪球（3D高斯椭球）
* 抛掷雪球（从3D投影到2D）
* 合成雪球（渲染成像）

##  3D Gaussian

### 3D高斯的物理意义

3D高斯具有很好的数学性质：

* 经过仿射变换后的的高斯核仍然闭合
* 3D高斯投影到2D后，依然为高斯分布

表达式：
$$
G(x)=\frac{1}{\sqrt{(2\pi)^{k}\vert \sum \vert}}e^{-\frac{1}{2}(x-\mu)^{T}\sum^{-1}(x-\mu)}
$$
其中$\sum$表示协方差矩阵，半正定，$\vert \sum \vert$为该矩阵的行列式

3D高斯函数的中心点由均值向量**$\mu$**决定，椭球体的三个主轴对应高斯分布的协方差矩阵$\sum$的特征向量，而特征值的大小则决定对应特征向量方向上的扩散程度**，特征值的平方根对应着主轴的长度。**

### 协方差矩阵的优化

**协方差矩阵只有在半正定时才有物理意义**，但是传统的梯度下降算法很难对矩阵施加此类约束，故不可以将协方差矩阵作为一个优化参数直接优化。

由协方差矩阵的几何意义可知，其表示该椭圆球在空间中的形状（缩放）和方向（旋转）。可通过特征值分解的方式将协方差矩阵进行分解
$$
\sum=Q\Lambda Q^{T}
$$

* 协方差矩阵$\sum$是一个3$\times$3的矩阵
* $Q$是由特征向量组成的正交矩阵（旋转矩阵）
* $\Lambda$是对角矩阵，其对角线上是协方差矩阵的三个特征值$\lambda_{1}$、$\lambda_{2}$、$\lambda_{3}$

由前文中所提到的主轴的长度对应着特征值的平方根，可对$\Lambda$进行进一步分解，就可得到原论文中的形式：
$$
\sum=RSS^{T}R^{T}
$$

* $S$是一个对角缩放矩阵，其对角线上是协方差矩阵的三个特征值的平方根$\sqrt{\lambda_{1}}$、$\sqrt{\lambda_{2}}、\sqrt{\lambda_{3}}$
* $R$是一个用四元数表示的旋转矩阵

通过优化旋转矩阵和缩放矩阵，可以保持协方差矩阵的半正定。

**而通过定义$R,S$以及均值$\mu$ 可得到三维空间中的所有三位高斯椭球**



