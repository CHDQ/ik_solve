# ik_solve

![python version](https://img.shields.io/badge/python-3.6%2B-blue)

# 使用说明

代码只在windows平台测试过

1. 文件说明
    1. `src->ik->ik_chain.py` 节点，ik链对象
    2. `src->ik->fabrik.py` 用fabrik算法解算ik链
    3. `src->ik->jacobian_ik.py` 用jacobian矩阵算法解算ik链
    4. `src->visible->visible.py` 用于实时显示ik解算效果
2. 进入`src->visible`目录，直接运行`python visible.py`文件即可
3. 移动effector键盘快捷键，数字键1,2,4,5,7,8在xyz不同轴的正负向移动effector

# 运行截图

![demo](https://github.com/CHDQ/ik_solve/blob/main/demo.gif)

# 待完成

    * jacobian 伪逆可能出现值过大情况，需要归一化
