# ik_solve

![python version](https://img.shields.io/badge/python-3.6%2B-blue)

# 使用说明

代码只在windows平台测试过

1. 文件说明
    1. `src->ik->ik_chain.py` 节点，ik链对象
    2. `src->ik->fabrik.py` 用fabrik算法解算ik链
    3. `src->ik->jacobian_ik.py` 用jacobian矩阵算法解算ik链 未实现角度约束
    4. `src->visible->visible.py` 用于实时显示ik解算效果
2. 进入`src->visible`目录，直接运行`python visible.py`文件即可
3. 移动effector键盘快捷键，数字键1,2,4,5,7,8在xyz不同轴的正负向移动effector

# 运行截图

![demo](https://github.com/CHDQ/ik_solve/blob/main/demo.gif)

# 遇到的问题
1. 开发初期，采用只记录骨骼朝向的方式计算jacobian矩阵，导致累积误差。实现流程如下

   1.骨骼朝向向量->2.四元数->3.旋转矩阵->4.欧拉角->5.jacobian->6.jacobian inverse->7.计算更新角度->8.转骨骼朝向->9.重复1

   上述循环，在1-3的过程中，由于每次迭代角度变化幅度很小，所以引起了误差。现象是迭代3轮以后，骨骼朝向不再跟随迭代轮次变化。
   
   解决方案

   迭代更新时，不再通过1-4的转换，而是直接记录角度变化，去掉转换过程中的误差
   
      
# 待完成
   
1. jacobian ik的任务优先级
2. jacobian ik角度约束

