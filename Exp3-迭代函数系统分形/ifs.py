import numpy as np
import matplotlib.pyplot as plt

def get_fern_params():
    """
    返回巴恩斯利蕨的IFS参数
    每个变换包含6个参数(a,b,c,d,e,f)和概率p
    """
    # TODO: 实现巴恩斯利蕨的参数
    return [
        # a     b      c      d      e     f      p
        [0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.01],   # 茎干
        [0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.85],  # 小叶
        [0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.07],  # 左大叶
        [-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.07]   # 右大叶
        ]

def get_tree_params():
    """
    返回概率树的IFS参数
    每个变换包含6个参数(a,b,c,d,e,f)和概率p
    """
    # TODO: 实现概率树的参数 
    return [
        # a     b      c      d      e     f      p
        [0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.10],   # 树干
        [0.42, -0.42, 0.42, 0.42, 0.00, 0.20, 0.45],  # 左分支
        [0.42, 0.42, -0.42, 0.42, 0.00, 0.20, 0.45]   # 右分支
        ]

def apply_transform(point, params):
    """
    应用单个变换到点
    :param point: 当前点坐标(x,y)
    :param params: 变换参数[a,b,c,d,e,f]
    :return: 变换后的新坐标(x',y')
    """
    x, y = point
    a, b, c, d, e, f = params
    x_new = a*x + b*y + e
    y_new = c*x + d*y + f
    return (x_new, y_new)


def run_ifs(ifs_params, num_points=100000, num_skip=100):
    """
    运行IFS迭代生成点集
    :param ifs_params: IFS参数列表
    :param num_points: 总点数
    :param num_skip: 跳过前n个点
    :return: 生成的点坐标数组
    """
    # TODO: 实现混沌游戏算法
    probs = [param[-1] for param in ifs_params]
    transforms = [param[:-1] for param in ifs_params]  # 去掉概率的变换参数
    
    # 初始化点
    x, y = 0.0, 0.0
    x_coords = []
    y_coords = []
    
    # 生成点序列
    for i in range(num_points + num_skip):
        # 随机选择变换
        idx = np.random.choice(len(transforms), p=probs)
        # 应用变换
        x, y = apply_transform((x, y), transforms[idx])
        # 跳过前num_skip个点
        if i >= num_skip:
            x_coords.append(x)
            y_coords.append(y)
    
    return np.array([x_coords, y_coords])

def plot_ifs(points, title="IFS Fractal"):
    """
    绘制IFS分形
    :param points: 点坐标数组
    :param title: 图像标题
    """
    # TODO: 实现分形绘制
    plt.figure(figsize=(8, 8))
    plt.scatter(points[0], points[1], s=0.1, c='green', alpha=0.5)
    plt.axis('equal')
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # 生成并绘制巴恩斯利蕨
    fern_params = get_fern_params()
    fern_points = run_ifs(fern_params)
    plot_ifs(fern_points, "Barnsley Fern")
    
    # 生成并绘制概率树
    tree_params = get_tree_params()
    tree_points = run_ifs(tree_params)
    plot_ifs(tree_points, "Probability Tree")
'''
        这段代码中return [
        # a     b      c      d      e     f      p
        [0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.01],   # 茎干
        [0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.85],  # 小叶
        [0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.07],  # 左大叶
        [-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.07]   # 右大叶
        ])
'''
