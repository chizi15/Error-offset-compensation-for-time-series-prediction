# 当y、yhat、ycomp是从所有单品中取值组成的序列时，当在第一大类中出现（y-ycomp==0），不能直接区分没有补偿和完全准确补偿的情况；
# 当在第二大类中出现（y-yhat==0），不能直接区分没有预测和完全准确预测的情况；
# 当在第三大类中出现（y-yhat==y-ycomp==0），不能直接区分没有预测、没有补偿、完全预测、完全补偿这四种情况；
# 基于上述原因，会导致无法准确统计某些分项的指标；建议y、yhat、ycomp从所有发生补偿的单品中取值，将没有发生补偿的单品排除。


import random
import pandas as pd
import numpy as np
import json

weights = []
for i in range(-99, 1):
    weights.append(0.94 ** i)

#  真实值组成的序列y、预测值组成的序列yhat、补偿值组成的序列ycomp，这三条序列组成中的单品是一一对应的，即同一单品在三条序列中的index相同
k1 = 5000
y_test = pd.Series(random.choices(range(0, 100), k=k1, weights=weights))  # 生成随机序列，模拟k1个单品在某一周的真实销量

z1 = pd.Series(np.sin(random.choices(range(0, 100), k=k1)))  # 假定这k1个是都有预测销量的单品
yhat_test = y_test + z1

z2 = pd.Series(np.sin(random.choices(range(0, 100), k=k1)))  # 并且假定这k1个是都发生补偿的单品
ycomp_test = y_test + z2


def cei(y, yhat, ycomp):
    # 得到经过第一层的三大类条件判断之后的真实值序列y_s1/2/3、预测值序列yhat_s1/2/3、补偿值序列ycomp_s1/2/3.

    # 1. 补偿变好这一大类：
    y_s1 = y[abs(y - yhat) > abs(y - ycomp)]  # 选出y序列中补偿变好的这一大类单品
    yhat_s1 = yhat[abs(y - yhat) > abs(y - ycomp)]  # 选出yhat序列中补偿变好的这一大类单品
    ycomp_s1 = ycomp[abs(y - yhat) > abs(y - ycomp)]  # 选出ycomp序列中补偿变好的这一大类单品

    # 2. 补偿变差这一大类：
    y_s2 = y[abs(y - yhat) < abs(y - ycomp)]  # 选出y序列中补偿变差的这一大类单品
    yhat_s2 = yhat[abs(y - yhat) < abs(y - ycomp)]  # 选出yhat序列中补偿变差的这一大类单品
    ycomp_s2 = ycomp[abs(y - yhat) < abs(y - ycomp)]  # 选出ycomp序列中补偿变差的这一大类单品

    # 3. 补偿无效果这一大类：分为1.发生补偿但补偿量恰好为零，2.恰好补偿到真实值的对侧等距点；
    # 其中1.又分为：1.1 真实值、预测值、补偿值恰好相等，1.2 预测值与补偿值相等，但与真实值不等。
    # 由于满足第三大类条件的单品很少，所以暂不对其进行细分。
    y_s3 = y[abs(y - yhat) == abs(y - ycomp)]  # 选出y序列中补偿无效果的这一大类单品

    if len(y) == len(yhat) == len(ycomp):  # 若该条件满足，则必有：len(y_s1)==len(yhat_s1)==len(ycomp_s1), \
        # len(y_s2)==len(yhat_s2)==len(ycomp_s2), len(y_s3)==len(yhat_s3)==len(ycomp_s3).
        # 因为这里只需获取三大类情况各自的单品数量m1/m2/m3，所以从y、yhat、ycomp任一序列中选取单品，得到的m1/m2/m3都相等。\
        # 下面是从y序列中选取单品，得到的yhat序列的长度m1/m2/m3.
        m1, m2, m3 = len(y_s1), len(y_s2), len(y_s3)
        mt = m1 + m2 + m3
        if mt == k1:
            x1, x2, x3 = m1 / mt * 100, m2 / mt * 100, m3 / mt * 100
        else:
            raise Exception('计算有误，变好、变化、无效果三种情况的单品数相加与发生补偿的单品总数不等。')
    else:
        raise Exception('真实值、预测值、补偿值组成的序列长度不等，建议化为等长，以避免后续某些指标代表的成分不唯一的情况出现。')

    # 在经过第一层条件判断之后，得到经过第二层条件判断的各个真实值序列、预测值序列、补偿值序列
    # 对于补偿变好这一大类中的6小类情况：
    # 1.1 预测补偿均低，补偿同侧变好：
    y_s1_c1 = y_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 > 0)]
    yhat_s1_c1 = yhat_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 > 0)]
    ycomp_s1_c1 = ycomp_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 > 0)]
    # 1.2 预测补偿均高，补偿同侧变好：
    y_s1_c2 = y_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 < 0)]
    yhat_s1_c2 = yhat_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 < 0)]
    ycomp_s1_c2 = ycomp_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 < 0)]
    # 1.3 预低补高，补偿对侧变好：
    y_s1_c3 = y_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 < 0)]
    yhat_s1_c3 = yhat_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 < 0)]
    ycomp_s1_c3 = ycomp_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 < 0)]
    # 1.4 预高补低，补偿对侧变好：
    y_s1_c4 = y_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 > 0)]
    yhat_s1_c4 = yhat_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 > 0)]
    ycomp_s1_c4 = ycomp_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 > 0)]
    # 1.5 预低，完全补偿：
    y_s1_c5 = y_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 == 0)]
    yhat_s1_c5 = yhat_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 == 0)]
    ycomp_s1_c5 = ycomp_s1[(y_s1 - yhat_s1 > 0) & (y_s1 - ycomp_s1 == 0)]
    # 1.6 预高，完全补偿：
    y_s1_c6 = y_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 == 0)]
    yhat_s1_c6 = yhat_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 == 0)]
    ycomp_s1_c6 = ycomp_s1[(y_s1 - yhat_s1 < 0) & (y_s1 - ycomp_s1 == 0)]

    # 对于补偿变差这一大类中的6小类情况：
    # 2.1 预测补偿均低，补偿同侧变差：
    y_s2_c1 = y_s2[(y_s2 - yhat_s2 > 0) & (y_s2 - ycomp_s2 > 0)]
    yhat_s2_c1 = yhat_s2[(y_s2 - yhat_s2 > 0) & (y_s2 - ycomp_s2 > 0)]
    ycomp_s2_c1 = ycomp_s2[(y_s2 - yhat_s2 > 0) & (y_s2 - ycomp_s2 > 0)]
    # 2.2 预测补偿均高，补偿同侧变差：
    y_s2_c2 = y_s2[(y_s2 - yhat_s2 < 0) & (y_s2 - ycomp_s2 < 0)]
    yhat_s2_c2 = yhat_s2[(y_s2 - yhat_s2 < 0) & (y_s2 - ycomp_s2 < 0)]
    ycomp_s2_c2 = ycomp_s2[(y_s2 - yhat_s2 < 0) & (y_s2 - ycomp_s2 < 0)]
    # 2.3 预低补高，补偿对侧变差：
    y_s2_c3 = y_s2[(y_s2 - yhat_s2 > 0) & (y_s2 - ycomp_s2 < 0)]
    yhat_s2_c3 = yhat_s2[(y_s2 - yhat_s2 > 0) & (y_s2 - ycomp_s2 < 0)]
    ycomp_s2_c3 = ycomp_s2[(y_s2 - yhat_s2 > 0) & (y_s2 - ycomp_s2 < 0)]
    # 2.4 预高补低，补偿对侧变差：
    y_s2_c4 = y_s2[(y_s2 - yhat_s2 < 0) & (y_s2 - ycomp_s2 > 0)]
    yhat_s2_c4 = yhat_s2[(y_s2 - yhat_s2 < 0) & (y_s2 - ycomp_s2 > 0)]
    ycomp_s2_c4 = ycomp_s2[(y_s2 - yhat_s2 < 0) & (y_s2 - ycomp_s2 > 0)]
    # 2.5 完全准确预测，补偿偏低：
    y_s2_c5 = y_s2[(y_s2 - yhat_s2 == 0) & (y_s2 - ycomp_s2 > 0)]
    yhat_s2_c5 = yhat_s2[(y_s2 - yhat_s2 == 0) & (y_s2 - ycomp_s2 > 0)]
    ycomp_s2_c5 = ycomp_s2[(y_s2 - yhat_s2 == 0) & (y_s2 - ycomp_s2 > 0)]
    # 2.6 完全准确预测，补偿偏高：
    y_s2_c6 = y_s2[(y_s2 - yhat_s2 == 0) & (y_s2 - ycomp_s2 < 0)]
    yhat_s2_c6 = yhat_s2[(y_s2 - yhat_s2 == 0) & (y_s2 - ycomp_s2 < 0)]
    ycomp_s2_c6 = ycomp_s2[(y_s2 - yhat_s2 == 0) & (y_s2 - ycomp_s2 < 0)]

    # 1. 对于第一大类即补偿变好的情况
    m11, m12 = len(y_s1_c1), len(y_s1_c2)  # 同侧补偿且变好
    m13, m14 = len(y_s1_c3), len(y_s1_c4)  # 对侧补偿且变好
    m15, m16 = len(y_s1_c5), len(y_s1_c6)  # 完全补偿
    m1syn, m1opp, m1com = m11 + m12, m13 + m14, m15 + m16  # 补偿变好时同侧、对侧、完全补偿的单品数
    if m11 + m12 + m13 + m14 + m15 + m16 == m1:
        x11, x12, x13, x14, x15, x16 = \
            m11 / m1 * 100, m12 / m1 * 100, m13 / m1 * 100, m14 / m1 * 100, m15 / m1 * 100, m16 / m1 * 100
        x1syn, x1opp, x1com = m1syn / m1 * 100, m1opp / m1 * 100, m1com / m1 * 100
    else:
        raise Exception('计算有误，当补偿变好时，其下6种子情况的单品总数相加与补偿变好的单品总数不等。')

    # 1.1 预测补偿均低，补偿同侧变好：
    pva_s1_c1 = abs(y_s1_c1 - yhat_s1_c1)  # 预测偏差量
    pva_s1_c1_sts = pva_s1_c1.describe()
    pvr_s1_c1 = abs(pva_s1_c1[y_s1_c1 != 0] / y_s1_c1[
        y_s1_c1 != 0]) * 100  # 预测偏差量与真实销量百分比，在此时的条件下，真实值(分母)为0的单品不能参与该指标计算，\
    # 否则会使下面describe的统计指标失真。所以pvr_s1_c1计算的是真实销量不为0的那些单品。
    pvr_s1_c1_sts = pvr_s1_c1.describe()

    cva_s1_c1 = abs(y_s1_c1 - ycomp_s1_c1)  # 补偿后剩余偏差量
    cva_s1_c1_sts = cva_s1_c1.describe()
    cvr_s1_c1 = abs(cva_s1_c1[y_s1_c1 != 0] / y_s1_c1[
        y_s1_c1 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比，在此时的条件下，真实值(分母)为0的单品不能参与该指标计算，\
    # 否则会使下面describe的统计指标失真。所以cvr_s1_c1计算的是真实销量不为0的那些单品。
    cvr_s1_c1_sts = cvr_s1_c1.describe()

    ca_s1_c1 = abs(yhat_s1_c1 - ycomp_s1_c1)  # 补偿量
    ca_s1_c1_sts = ca_s1_c1.describe()
    car_s1_c1 = abs(ca_s1_c1[pva_s1_c1 != 0] / pva_s1_c1[pva_s1_c1 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s1_c1_sts = car_s1_c1.describe()

    if (len(pva_s1_c1) == len(cva_s1_c1) == len(ca_s1_c1) == len(car_s1_c1) == m11) & \
            (len(pva_s1_c1) >= len(pvr_s1_c1)) & (len(cva_s1_c1) >= len(cvr_s1_c1)):
        pass
    else:
        raise Exception('在1.1即预测补偿均低，补偿同侧变好的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 1.2 预测补偿均高，补偿同侧变好：
    pva_s1_c2 = abs(y_s1_c2 - yhat_s1_c2)  # 预测偏差量
    pva_s1_c2_sts = pva_s1_c2.describe()
    pvr_s1_c2 = abs(pva_s1_c2[y_s1_c2 != 0] / y_s1_c2[y_s1_c2 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s1_c2_sts = pvr_s1_c2.describe()

    cva_s1_c2 = abs(y_s1_c2 - ycomp_s1_c2)  # 补偿后剩余偏差量
    cva_s1_c2_sts = cva_s1_c2.describe()
    cvr_s1_c2 = abs(cva_s1_c2[y_s1_c2 != 0] / y_s1_c2[y_s1_c2 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s1_c2_sts = cvr_s1_c2.describe()

    ca_s1_c2 = abs(yhat_s1_c2 - ycomp_s1_c2)  # 补偿量
    ca_s1_c2_sts = ca_s1_c2.describe()
    car_s1_c2 = abs(ca_s1_c2[pva_s1_c2 != 0] / pva_s1_c2[pva_s1_c2 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s1_c2_sts = car_s1_c2.describe()

    if (len(pva_s1_c2) == len(cva_s1_c2) == len(ca_s1_c2) == len(car_s1_c2) == m12) & \
            (len(pva_s1_c2) >= len(pvr_s1_c2)) & (len(cva_s1_c2) >= len(cvr_s1_c2)):
        pass
    else:
        raise Exception('在1.2即预测补偿均高，补偿同侧变好的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 1.3 预测偏低补偿偏高，补偿对侧变好：
    pva_s1_c3 = abs(y_s1_c3 - yhat_s1_c3)  # 预测偏差量
    pva_s1_c3_sts = pva_s1_c3.describe()
    pvr_s1_c3 = abs(pva_s1_c3[y_s1_c3 != 0] / y_s1_c3[y_s1_c3 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s1_c3_sts = pvr_s1_c3.describe()

    cva_s1_c3 = abs(y_s1_c3 - ycomp_s1_c3)  # 补偿后剩余偏差量
    cva_s1_c3_sts = cva_s1_c3.describe()
    cvr_s1_c3 = abs(cva_s1_c3[y_s1_c3 != 0] / y_s1_c3[y_s1_c3 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s1_c3_sts = cvr_s1_c3.describe()

    ca_s1_c3 = abs(yhat_s1_c3 - ycomp_s1_c3)  # 补偿量
    ca_s1_c3_sts = ca_s1_c3.describe()
    car_s1_c3 = abs(ca_s1_c3[pva_s1_c3 != 0] / pva_s1_c3[pva_s1_c3 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s1_c3_sts = car_s1_c3.describe()

    if (len(pva_s1_c3) == len(cva_s1_c3) == len(ca_s1_c3) == len(car_s1_c3) == m13) & \
            (len(pva_s1_c3) >= len(pvr_s1_c3)) & (len(cva_s1_c3) >= len(cvr_s1_c3)):
        pass
    else:
        raise Exception('在1.3即预测偏低补偿偏高，补偿对侧变好的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 1.4 预测偏高补偿偏低，补偿对侧变好：
    pva_s1_c4 = abs(y_s1_c4 - yhat_s1_c4)  # 预测偏差量
    pva_s1_c4_sts = pva_s1_c4.describe()
    pvr_s1_c4 = abs(pva_s1_c4[y_s1_c4 != 0] / y_s1_c4[y_s1_c4 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s1_c4_sts = pvr_s1_c4.describe()

    cva_s1_c4 = abs(y_s1_c4 - ycomp_s1_c4)  # 补偿后剩余偏差量
    cva_s1_c4_sts = cva_s1_c4.describe()
    cvr_s1_c4 = abs(cva_s1_c4[y_s1_c4 != 0] / y_s1_c4[y_s1_c4 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s1_c4_sts = cvr_s1_c4.describe()

    ca_s1_c4 = abs(yhat_s1_c4 - ycomp_s1_c4)  # 补偿量
    ca_s1_c4_sts = ca_s1_c4.describe()
    car_s1_c4 = abs(ca_s1_c4[pva_s1_c4 != 0] / pva_s1_c4[pva_s1_c4 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s1_c4_sts = car_s1_c4.describe()

    if (len(pva_s1_c4) == len(cva_s1_c4) == len(ca_s1_c4) == len(car_s1_c4) == m14) & \
            (len(pva_s1_c4) >= len(pvr_s1_c4)) & (len(cva_s1_c4) >= len(cvr_s1_c4)):
        pass
    else:
        raise Exception('在1.4即预测偏高补偿偏低，补偿对侧变好的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 1.5 预测偏低，完全补偿：
    pva_s1_c5 = abs(y_s1_c5 - yhat_s1_c5)  # 预测偏差量
    pva_s1_c5_sts = pva_s1_c5.describe()
    pvr_s1_c5 = abs(pva_s1_c5[y_s1_c5 != 0] / y_s1_c5[y_s1_c5 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s1_c5_sts = pvr_s1_c5.describe()

    cva_s1_c5 = abs(y_s1_c5 - ycomp_s1_c5)  # 补偿后剩余偏差量
    cva_s1_c5_sts = cva_s1_c5.describe()
    cvr_s1_c5 = abs(cva_s1_c5[y_s1_c5 != 0] / y_s1_c5[y_s1_c5 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s1_c5_sts = cvr_s1_c5.describe()
    if ((cva_s1_c5 == 0).sum() == len(cva_s1_c5)) & ((cvr_s1_c5 == 0).sum() == len(cvr_s1_c5)):
        pass
    else:
        raise Exception('1.5 在第一大类的第五小类中，即预测偏低，完全补偿这种情况下，\
    补偿偏差量(cva_s1_c5)和/或补偿偏差比(cvr_s1_c5)存在计算有误的情况，请检查错误原因。')

    ca_s1_c5 = abs(yhat_s1_c5 - ycomp_s1_c5)  # 补偿量
    ca_s1_c5_sts = ca_s1_c5.describe()
    car_s1_c5 = abs(ca_s1_c5[pva_s1_c5 != 0] / pva_s1_c5[pva_s1_c5 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s1_c5_sts = car_s1_c5.describe()
    if ((ca_s1_c5 == pva_s1_c5).sum() == len(ca_s1_c5)) & ((car_s1_c5 == 100).sum() == len(car_s1_c5)):
        pass
    else:
        raise Exception('1.5 在第一大类的第五小类中，即预测偏低，完全补偿这种情况下，\
    补偿量(ca_s1_c5)、补偿比(car_s1_c5)、预测偏差量(pva_s1_c5)中存在计算有误的情况，请检查错误原因。')

    if (len(pva_s1_c5) == len(cva_s1_c5) == len(ca_s1_c5) == len(car_s1_c5) == m15) & \
            (len(pva_s1_c5) >= len(pvr_s1_c5)) & (len(cva_s1_c5) >= len(cvr_s1_c5)):
        pass
    else:
        raise Exception('在1.5即预测偏低完全补偿的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 1.6 预测偏高，完全补偿：
    pva_s1_c6 = abs(y_s1_c6 - yhat_s1_c6)  # 预测偏差量
    pva_s1_c6_sts = pva_s1_c6.describe()
    pvr_s1_c6 = abs(pva_s1_c6[y_s1_c6 != 0] / y_s1_c6[y_s1_c6 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s1_c6_sts = pvr_s1_c6.describe()

    cva_s1_c6 = abs(y_s1_c6 - ycomp_s1_c6)  # 补偿后剩余偏差量
    cva_s1_c6_sts = cva_s1_c6.describe()
    cvr_s1_c6 = abs(cva_s1_c6[y_s1_c6 != 0] / y_s1_c6[y_s1_c6 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s1_c6_sts = cvr_s1_c6.describe()
    if ((cva_s1_c6 == 0).sum() == len(cva_s1_c6)) & ((cvr_s1_c6 == 0).sum() == len(cvr_s1_c6)):
        pass
    else:
        raise Exception('1.6 在第一大类的第六小类中，即预测偏高，完全补偿这种情况下，\
    补偿偏差量(cva_s1_c6)和/或补偿偏差比(cvr_s1_c6)存在计算有误的情况，请检查错误原因。')

    ca_s1_c6 = abs(yhat_s1_c6 - ycomp_s1_c6)  # 补偿量
    ca_s1_c6_sts = ca_s1_c6.describe()
    car_s1_c6 = abs(ca_s1_c6[pva_s1_c6 != 0] / pva_s1_c6[pva_s1_c6 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s1_c6_sts = car_s1_c6.describe()
    if ((ca_s1_c6 == pva_s1_c6).sum() == len(ca_s1_c6)) & ((car_s1_c6 == 100).sum() == len(car_s1_c6)):
        pass
    else:
        raise Exception('1.6 在第一大类的第六小类中，即预测偏高，完全补偿这种情况下，\
    补偿量(ca_s1_c6)、补偿比(car_s1_c6)、预测偏差量(pva_s1_c6)中存在计算有误的情况，请检查错误原因。')

    if (len(pva_s1_c6) == len(cva_s1_c6) == len(ca_s1_c6) == len(car_s1_c6) == m16) & \
            (len(pva_s1_c6) >= len(pvr_s1_c6)) & (len(cva_s1_c6) >= len(cvr_s1_c6)):
        pass
    else:
        raise Exception('在1.6即预测偏高完全补偿的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 2. 对于第二大类即补偿变差的情况
    m21, m22 = len(y_s2_c1), len(y_s2_c2)  # 同侧补偿且变差
    m23, m24 = len(y_s2_c3), len(y_s2_c4)  # 对侧补偿且变差
    m25, m26 = len(y_s2_c5), len(y_s2_c6)  # 完全准确预测
    m2syn, m2opp, m2com = m21 + m22, m23 + m14, m25 + m26  # 补偿变差时同侧、对侧、完全准确预测的单品数
    if m21 + m22 + m23 + m24 + m25 + m26 == m2:
        x21, x22, x23, x24, x25, x26 = \
            m21 / m2 * 100, m22 / m2 * 100, m23 / m2 * 100, m24 / m2 * 100, m25 / m2 * 100, m26 / m2 * 100
        x2syn, x2opp, x2com = m2syn / m2 * 100, m2opp / m2 * 100, m2com / m2 * 100
    else:
        raise Exception('计算有误，当补偿变差时，其下6种子情况的单品总数相加与补偿变差的单品总数不等。')

    # 2.1 预测补偿均低，补偿同侧变差：
    pva_s2_c1 = abs(y_s2_c1 - yhat_s2_c1)  # 预测偏差量
    pva_s2_c1_sts = pva_s2_c1.describe()
    pvr_s2_c1 = abs(pva_s2_c1[y_s2_c1 != 0] / y_s2_c1[y_s2_c1 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s2_c1_sts = pvr_s2_c1.describe()

    cva_s2_c1 = abs(y_s2_c1 - ycomp_s2_c1)  # 补偿后剩余偏差量
    cva_s2_c1_sts = cva_s2_c1.describe()
    cvr_s2_c1 = abs(cva_s2_c1[y_s2_c1 != 0] / y_s2_c1[y_s2_c1 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s2_c1_sts = cvr_s2_c1.describe()

    ca_s2_c1 = abs(yhat_s2_c1 - ycomp_s2_c1)  # 补偿量
    ca_s2_c1_sts = ca_s2_c1.describe()
    car_s2_c1 = abs(ca_s2_c1[pva_s2_c1 != 0] / pva_s2_c1[pva_s2_c1 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s2_c1_sts = car_s2_c1.describe()

    if (len(pva_s2_c1) == len(cva_s2_c1) == len(ca_s2_c1) == len(car_s2_c1) == m21) & \
            (len(pva_s2_c1) >= len(pvr_s2_c1)) & (len(cva_s2_c1) >= len(cvr_s2_c1)):
        pass
    else:
        raise Exception('在2.1即预测补偿均低，补偿同侧变差的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 2.2 预测补偿均高，补偿同侧变差：
    pva_s2_c2 = abs(y_s2_c2 - yhat_s2_c2)  # 预测偏差量
    pva_s2_c2_sts = pva_s2_c2.describe()
    pvr_s2_c2 = abs(pva_s2_c2[y_s2_c2 != 0] / y_s2_c2[y_s2_c2 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s2_c2_sts = pvr_s2_c2.describe()

    cva_s2_c2 = abs(y_s2_c2 - ycomp_s2_c2)  # 补偿后剩余偏差量
    cva_s2_c2_sts = cva_s2_c2.describe()
    cvr_s2_c2 = abs(cva_s2_c2[y_s2_c2 != 0] / y_s2_c2[y_s2_c2 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s2_c2_sts = cvr_s2_c2.describe()

    ca_s2_c2 = abs(yhat_s2_c2 - ycomp_s2_c2)  # 补偿量
    ca_s2_c2_sts = ca_s2_c2.describe()
    car_s2_c2 = abs(ca_s2_c2[pva_s2_c2 != 0] / pva_s2_c2[pva_s2_c2 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s2_c2_sts = car_s2_c2.describe()

    if (len(pva_s2_c2) == len(cva_s2_c2) == len(ca_s2_c2) == len(car_s2_c2) == m22) & \
            (len(pva_s2_c2) >= len(pvr_s2_c2)) & (len(cva_s2_c2) >= len(cvr_s2_c2)):
        pass
    else:
        raise Exception('在2.2即预测补偿均高，补偿同侧变差的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 2.3 预测偏低补偿偏高，补偿对侧变差：
    pva_s2_c3 = abs(y_s2_c3 - yhat_s2_c3)  # 预测偏差量
    pva_s2_c3_sts = pva_s2_c3.describe()
    pvr_s2_c3 = abs(pva_s2_c3[y_s2_c3 != 0] / y_s2_c3[y_s2_c3 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s2_c3_sts = pvr_s2_c3.describe()

    cva_s2_c3 = abs(y_s2_c3 - ycomp_s2_c3)  # 补偿后剩余偏差量
    cva_s2_c3_sts = cva_s2_c3.describe()
    cvr_s2_c3 = abs(cva_s2_c3[y_s2_c3 != 0] / y_s2_c3[y_s2_c3 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s2_c3_sts = cvr_s2_c3.describe()

    ca_s2_c3 = abs(yhat_s2_c3 - ycomp_s2_c3)  # 补偿量
    ca_s2_c3_sts = ca_s2_c3.describe()
    car_s2_c3 = abs(ca_s2_c3[pva_s2_c3 != 0] / pva_s2_c3[pva_s2_c3 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s2_c3_sts = car_s2_c3.describe()

    if (len(pva_s2_c3) == len(cva_s2_c3) == len(ca_s2_c3) == len(car_s2_c3) == m23) & \
            (len(pva_s2_c3) >= len(pvr_s2_c3)) & (len(cva_s2_c3) >= len(cvr_s2_c3)):
        pass
    else:
        raise Exception('在2.3即预测偏低补偿偏高，补偿对侧变差的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 2.4 预测偏高补偿偏低，补偿对侧变差：
    pva_s2_c4 = abs(y_s2_c4 - yhat_s2_c4)  # 预测偏差量
    pva_s2_c4_sts = pva_s2_c4.describe()
    pvr_s2_c4 = abs(pva_s2_c4[y_s2_c4 != 0] / y_s2_c4[y_s2_c4 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s2_c4_sts = pvr_s2_c4.describe()

    cva_s2_c4 = abs(y_s2_c4 - ycomp_s2_c4)  # 补偿后剩余偏差量
    cva_s2_c4_sts = cva_s2_c4.describe()
    cvr_s2_c4 = abs(cva_s2_c4[y_s2_c4 != 0] / y_s2_c4[y_s2_c4 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s2_c4_sts = cvr_s2_c4.describe()

    ca_s2_c4 = abs(yhat_s2_c4 - ycomp_s2_c4)  # 补偿量
    ca_s2_c4_sts = ca_s2_c4.describe()
    car_s2_c4 = abs(ca_s2_c4[pva_s2_c4 != 0] / pva_s2_c4[pva_s2_c4 != 0]) * 100  # 补偿量占预测偏差量百分比
    car_s2_c4_sts = car_s2_c4.describe()

    if (len(pva_s2_c4) == len(cva_s2_c4) == len(ca_s2_c4) == len(car_s2_c4) == m24) & \
            (len(pva_s2_c4) >= len(pvr_s2_c4)) & (len(cva_s2_c4) >= len(cvr_s2_c4)):
        pass
    else:
        raise Exception('在2.4即预测偏高补偿偏低，补偿对侧变差的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 2.5 完全准确预测，补偿偏低：
    pva_s2_c5 = abs(y_s2_c5 - yhat_s2_c5)  # 预测偏差量
    pva_s2_c5_sts = pva_s2_c5.describe()
    pvr_s2_c5 = abs(pva_s2_c5[y_s2_c5 != 0] / y_s2_c5[y_s2_c5 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s2_c5_sts = pvr_s2_c5.describe()
    if ((pva_s2_c5 == 0).sum() == len(pva_s2_c5)) & ((pvr_s2_c5 == 0).sum() == len(pvr_s2_c5)):
        pass
    else:
        raise Exception('2.5 在第二大类的第五小类中，即完全准确预测补偿偏低这种情况下，\
    预测偏差量(pva_s2_c5)和/或预测偏差比(pvr_s2_c5)存在计算有误的情况，请检查错误原因。')

    cva_s2_c5 = abs(y_s2_c5 - ycomp_s2_c5)  # 补偿后剩余偏差量
    cva_s2_c5_sts = cva_s2_c5.describe()
    cvr_s2_c5 = abs(cva_s2_c5[y_s2_c5 != 0] / y_s2_c5[y_s2_c5 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s2_c5_sts = cvr_s2_c5.describe()

    ca_s2_c5 = abs(yhat_s2_c5 - ycomp_s2_c5)  # 补偿量
    ca_s2_c5_sts = ca_s2_c5.describe()
    car_s2_c5_reciprocal = pva_s2_c5 / ca_s2_c5  # 因为此处补偿量占预测偏差量百分比为无穷大，所以采用其倒数，即预测偏差量占补偿量百分比，其结果均为0，避免报错
    car_s2_c5_reciprocal_sts = car_s2_c5_reciprocal.describe()
    if ((ca_s2_c5 == cva_s2_c5).sum() == len(ca_s2_c5)) & \
            ((car_s2_c5_reciprocal == 0).sum() == len(car_s2_c5_reciprocal)):
        pass
    else:
        raise Exception('2.5 在第二大类的第五小类中，即预测完全准确，补偿偏低这种情况下，\
    补偿量(ca_s2_c5)、补偿比的倒数(car_s2_c5_reciprocal)、补偿后剩余偏差量(cva_s2_c5)中存在计算有误的情况，请检查错误原因。')

    if (len(pva_s2_c5) == len(cva_s2_c5) == len(ca_s2_c5) == len(car_s2_c5_reciprocal) == m25) & \
            (len(pva_s2_c5) >= len(pvr_s2_c5)) & (len(cva_s2_c5) >= len(cvr_s2_c5)):
        pass
    else:
        raise Exception('在2.5即完全准确预测，补偿偏低的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 2.6 完全准确预测，补偿偏高：
    pva_s2_c6 = abs(y_s2_c6 - yhat_s2_c6)  # 预测偏差量
    pva_s2_c6_sts = pva_s2_c6.describe()
    pvr_s2_c6 = abs(pva_s2_c6[y_s2_c6 != 0] / y_s2_c6[y_s2_c6 != 0]) * 100  # 预测偏差量与真实销量百分比
    pvr_s2_c6_sts = pvr_s2_c6.describe()
    if ((pva_s2_c6 == 0).sum() == len(pva_s2_c6)) & ((pvr_s2_c6 == 0).sum() == len(pvr_s2_c6)):
        pass
    else:
        raise Exception('2.5 在第二大类的第六小类中，即完全准确预测补偿偏高这种情况下，\
    预测偏差量(pva_s2_c6)和/或预测偏差比(pvr_s2_c6)存在计算有误的情况，请检查错误原因。')

    cva_s2_c6 = abs(y_s2_c6 - ycomp_s2_c6)  # 补偿后剩余偏差量
    cva_s2_c6_sts = cva_s2_c6.describe()
    cvr_s2_c6 = abs(cva_s2_c6[y_s2_c6 != 0] / y_s2_c6[y_s2_c6 != 0]) * 100  # 补偿后剩余偏差量与真实销量百分比
    cvr_s2_c6_sts = cvr_s2_c6.describe()

    ca_s2_c6 = abs(yhat_s2_c6 - ycomp_s2_c6)  # 补偿量
    ca_s2_c6_sts = ca_s2_c6.describe()
    car_s2_c6_reciprocal = pva_s2_c6 / ca_s2_c6  # 因为此处补偿量占预测偏差量百分比为无穷大，所以采用其倒数，即预测偏差量占补偿量百分比，其结果均为0，避免报错
    car_s2_c6_reciprocal_sts = car_s2_c6_reciprocal.describe()
    if ((ca_s2_c6 == cva_s2_c6).sum() == len(ca_s2_c6)) & \
            ((car_s2_c6_reciprocal == 0).sum() == len(car_s2_c6_reciprocal)):
        pass
    else:
        raise Exception('2.6 在第二大类的第六小类中，即预测完全准确，补偿偏高这种情况下，\
    补偿量(ca_s2_c6)、补偿比的倒数(car_s2_c6_reciprocal)、补偿后剩余偏差量(cva_s2_c6)中存在计算有误的情况，请检查错误原因。')

    if (len(pva_s2_c6) == len(cva_s2_c6) == len(ca_s2_c6) == len(car_s2_c6_reciprocal) == m26) & \
            (len(pva_s2_c6) >= len(pvr_s2_c6)) & (len(cva_s2_c6) >= len(cvr_s2_c6)):
        pass
    else:
        raise Exception('在2.6即完全准确预测，补偿偏高的情况中，各分量的长度匹配存在错误，请检查该异常。')

    # 共得到6+18+6*6+18+6*6=114种指标变量
    return mt, m1, x1, m2, x2, m3, x3, \
        m11, x11, m12, x12, m13, x13, m14, x14, m15, x15, m16, x16, m1syn, x1syn, m1opp, x1opp, m1com, x1com, \
        pva_s1_c1_sts, pvr_s1_c1_sts, cva_s1_c1_sts, cvr_s1_c1_sts, ca_s1_c1_sts, car_s1_c1_sts, \
        pva_s1_c2_sts, pvr_s1_c2_sts, cva_s1_c2_sts, cvr_s1_c2_sts, ca_s1_c2_sts, car_s1_c2_sts, \
        pva_s1_c3_sts, pvr_s1_c3_sts, cva_s1_c3_sts, cvr_s1_c3_sts, ca_s1_c3_sts, car_s1_c3_sts, \
        pva_s1_c4_sts, pvr_s1_c4_sts, cva_s1_c4_sts, cvr_s1_c4_sts, ca_s1_c4_sts, car_s1_c4_sts, \
        pva_s1_c5_sts, pvr_s1_c5_sts, cva_s1_c5_sts, cvr_s1_c5_sts, ca_s1_c5_sts, car_s1_c5_sts, \
        pva_s1_c6_sts, pvr_s1_c6_sts, cva_s1_c6_sts, cvr_s1_c6_sts, ca_s1_c6_sts, car_s1_c6_sts, \
        m21, x21, m22, x22, m23, x23, m24, x24, m25, x25, m26, x26, m2syn, x2syn, m2opp, x2opp, m2com, x2com, \
        pva_s2_c1_sts, pvr_s2_c1_sts, cva_s2_c1_sts, cvr_s2_c1_sts, ca_s2_c1_sts, car_s2_c1_sts, \
        pva_s2_c2_sts, pvr_s2_c2_sts, cva_s2_c2_sts, cvr_s2_c2_sts, ca_s2_c2_sts, car_s2_c2_sts, \
        pva_s2_c3_sts, pvr_s2_c3_sts, cva_s2_c3_sts, cvr_s2_c3_sts, ca_s2_c3_sts, car_s2_c3_sts, \
        pva_s2_c4_sts, pvr_s2_c4_sts, cva_s2_c4_sts, cvr_s2_c4_sts, ca_s2_c4_sts, car_s2_c4_sts, \
        pva_s2_c5_sts, pvr_s2_c5_sts, cva_s2_c5_sts, cvr_s2_c5_sts, ca_s2_c5_sts, car_s2_c5_reciprocal_sts, \
        pva_s2_c6_sts, pvr_s2_c6_sts, cva_s2_c6_sts, cvr_s2_c6_sts, ca_s2_c6_sts, car_s2_c6_reciprocal_sts


a = cei(y=y_test, yhat=yhat_test, ycomp=ycomp_test)

result1 = \
    {'第一层条件，判断总体补偿情况':
        {'本次统计发生补偿的单品总数': a[0],
         '补偿变好的单品数': a[1],
         '变好单品占全部发生补偿的单品百分比': a[2],
         '补偿变差的单品数': a[3],
         '变差单品占全部发生补偿的单品百分比': a[4],
         '补偿无效果的单品数': a[5],
         '无效果单品占全部发生补偿的单品百分比': a[6]
         },
     '第一大类情况，即补偿变好时':
        {'第一大类中六个子类的总体指标':
            {'预测销量与补偿后销量均比真实销量低，补偿变好且发生在同侧，该种情况的单品数': a[7],
             '均低同侧变好占补偿变好单品数的百分比': a[8],
             '预测销量与补偿后销量均比真实销量高，补偿变好且发生在同侧，该种情况的单品数': a[9],
             '均高同侧变好占补偿变好单品数的百分比': a[10],
             '预测销量偏低，补偿后销量偏高，补偿变好且发生在对侧，该种情况的单品数': a[11],
             '预低补高对侧变好占补偿变好单品数的百分比': a[12],
             '预测销量偏高，补偿后销量偏低，补偿变好且发生在对侧，该种情况的单品数': a[13],
             '预高补低对侧变好占补偿变好单品数的百分比': a[14],
             '预测销量偏低，完全补偿，该种情况的单品数': a[15],
             '预低全补占补偿变好单品数的百分比': a[16],
             '预测销量偏高，完全补偿，该种情况的单品数': a[17],
             '预高全补占补偿变好单品数的百分比': a[18],
             '补偿变好且发生在同侧的单品数': a[19],
             '补好同侧单品数占补偿变好单品数的百分比': a[20],
             '补偿变好且发生在对侧的单品数': a[21],
             '补好对侧单品数占补偿变好单品数的百分比': a[22],
             '完全补偿的单品数': a[23],
             '完全补偿的单品数占补偿变好单品数的百分比': a[24]
             },
         '1.1 预测补偿均低，补偿同侧变好时各项指标':
            {'1.1中预测偏差量的统计指标': a[25],
             '1.1中预测偏差量与真实销量百分比的统计指标': a[26],
             '1.1中补偿后剩余偏差量的统计指标': a[27],
             '1.1中补偿后剩余偏差量与真实销量百分比的统计指标': a[28],
             '1.1中补偿量的统计指标': a[29],
             '1.1中补偿量占预测偏差量百分比的统计指标': a[30]
             },
         '1.2 预测补偿均高，补偿同侧变好时各项指标':
            {'1.2中预测偏差量的统计指标': a[31],
             '1.2中预测偏差量与真实销量百分比的统计指标': a[32],
             '1.2中补偿后剩余偏差量的统计指标': a[33],
             '1.2中补偿后剩余偏差量与真实销量百分比的统计指标': a[34],
             '1.2中补偿量的统计指标': a[35],
             '1.2中补偿量占预测偏差量百分比的统计指标': a[36]
             },
         '1.3 预测偏低补偿偏高，补偿对侧变好时各项指标':
            {'1.3中预测偏差量的统计指标': a[37],
             '1.3中预测偏差量与真实销量百分比的统计指标': a[38],
             '1.3中补偿后剩余偏差量的统计指标': a[39],
             '1.3中补偿后剩余偏差量与真实销量百分比的统计指标': a[40],
             '1.3中补偿量的统计指标': a[41],
             '1.3中补偿量占预测偏差量百分比的统计指标': a[42]
             },
         '1.4 预测偏高补偿偏低，补偿对侧变好时各项指标':
            {'1.4中预测偏差量的统计指标': a[43],
             '1.4中预测偏差量与真实销量百分比的统计指标': a[44],
             '1.4中补偿后剩余偏差量的统计指标': a[45],
             '1.4中补偿后剩余偏差量与真实销量百分比的统计指标': a[46],
             '1.4中补偿量的统计指标': a[47],
             '1.4中补偿量占预测偏差量百分比的统计指标': a[48]
             },
         '1.5 预测偏低，完全补偿时各项指标':
            {'1.5中预测偏差量的统计指标': a[49],
             '1.5中预测偏差量与真实销量百分比的统计指标': a[50],
             '1.5中补偿后剩余偏差量的统计指标': a[51],
             '1.5中补偿后剩余偏差量与真实销量百分比的统计指标': a[52],
             '1.5中补偿量的统计指标': a[53],
             '1.5中补偿量占预测偏差量百分比的统计指标': a[54]
             },
         '1.6 预测偏高，完全补偿时各项指标':
            {'1.6中预测偏差量的统计指标': a[55],
             '1.6中预测偏差量与真实销量百分比的统计指标': a[56],
             '1.6中补偿后剩余偏差量的统计指标': a[57],
             '1.6中补偿后剩余偏差量与真实销量百分比的统计指标': a[58],
             '1.6中补偿量的统计指标': a[59],
             '1.6中补偿量占预测偏差量百分比的统计指标': a[60]
             },
         },
     '第二大类情况，即补偿变差时':
        {'第二大类中六个子类的总体指标':
            {'预测销量与补偿后销量均比真实销量低，补偿变差且发生在同侧，该种情况的单品数': a[61],
             '均低同侧变差占补偿变差单品数的百分比': a[62],
             '预测销量与补偿后销量均比真实销量高，补偿变差且发生在同侧，该种情况的单品数': a[63],
             '均高同侧变差占补偿变差单品数的百分比': a[64],
             '预测销量偏低，补偿后销量偏高，补偿变差且发生在对侧，该种情况的单品数': a[65],
             '预低补高对侧变差占补偿变差单品数的百分比': a[66],
             '预测销量偏高，补偿后销量偏低，补偿变差且发生在对侧，该种情况的单品数': a[67],
             '预高补低对侧变差占补偿变差单品数的百分比': a[68],
             '预测销量偏低，完全补偿，该种情况的单品数': a[69],
             '预低全补占补偿变差单品数的百分比': a[70],
             '预测销量偏高，完全补偿，该种情况的单品数': a[71],
             '预高全补占补偿变差单品数的百分比': a[72],
             '补偿变差且发生在同侧的单品数': a[73],
             '补好同侧单品数占补偿变差单品数的百分比': a[74],
             '补偿变差且发生在对侧的单品数': a[75],
             '补好对侧单品数占补偿变差单品数的百分比': a[76],
             '完全补偿的单品数': a[77],
             '完全补偿的单品数占补偿变差单品数的百分比': a[78]
             },
         '2.1 预测补偿均低，补偿同侧变差时各项指标':
            {'2.1中预测偏差量的统计指标': a[79],
             '2.1中预测偏差量与真实销量百分比的统计指标': a[80],
             '2.1中补偿后剩余偏差量的统计指标': a[81],
             '2.1中补偿后剩余偏差量与真实销量百分比的统计指标': a[82],
             '2.1中补偿量的统计指标': a[83],
             '2.1中补偿量占预测偏差量百分比的统计指标': a[84]
             },
         '2.2 预测补偿均高，补偿同侧变差时各项指标':
            {'2.2中预测偏差量的统计指标': a[85],
             '2.2中预测偏差量与真实销量百分比的统计指标': a[86],
             '2.2中补偿后剩余偏差量的统计指标': a[87],
             '2.2中补偿后剩余偏差量与真实销量百分比的统计指标': a[88],
             '2.2中补偿量的统计指标': a[89],
             '2.2中补偿量占预测偏差量百分比的统计指标': a[90]
             },
         '2.3 预测偏低补偿偏高，补偿对侧变差时各项指标':
            {'2.3中预测偏差量的统计指标': a[91],
             '2.3中预测偏差量与真实销量百分比的统计指标': a[92],
             '2.3中补偿后剩余偏差量的统计指标': a[93],
             '2.3中补偿后剩余偏差量与真实销量百分比的统计指标': a[94],
             '2.3中补偿量的统计指标': a[95],
             '2.3中补偿量占预测偏差量百分比的统计指标': a[96]
             },
         '2.4 预测偏高补偿偏低，补偿对侧变差时各项指标':
            {'2.4中预测偏差量的统计指标': a[97],
             '2.4中预测偏差量与真实销量百分比的统计指标': a[98],
             '2.4中补偿后剩余偏差量的统计指标': a[99],
             '2.4中补偿后剩余偏差量与真实销量百分比的统计指标': a[100],
             '2.4中补偿量的统计指标': a[101],
             '2.4中补偿量占预测偏差量百分比的统计指标': a[102]
             },
         '2.5 完全准确预测，补偿偏低时各项指标':
            {'2.5中预测偏差量的统计指标': a[103],
             '2.5中预测偏差量与真实销量百分比的统计指标': a[104],
             '2.5中补偿后剩余偏差量的统计指标': a[105],
             '2.5中补偿后剩余偏差量与真实销量百分比的统计指标': a[106],
             '2.5中补偿量的统计指标': a[107],
             '2.5中预测偏差量占补偿量百分比的统计指标': a[108]
             },
         '2.6 完全准确预测，补偿偏高时各项指标':
            {'2.6中预测偏差量的统计指标': a[109],
             '2.6中预测偏差量与真实销量百分比的统计指标': a[110],
             '2.6中补偿后剩余偏差量的统计指标': a[111],
             '2.6中补偿后剩余偏差量与真实销量百分比的统计指标': a[112],
             '2.6中补偿量的统计指标': a[113],
             '2.6中预测偏差量占补偿量百分比的统计指标': a[114]
             }
         }
     }

# 输出json格式结果，方便运行代码后立即查看。
result2 = \
    {'第一层条件，判断总体补偿情况':
        ['本次统计发生补偿的单品总数', a[0],
         '补偿变好的单品数', a[1],
         '变好单品占全部发生补偿的单品百分比', a[2],
         '补偿变差的单品数', a[3],
         '变差单品占全部发生补偿的单品百分比', a[4],
         '补偿无效果的单品数', a[5],
         '无效果单品占全部发生补偿的单品百分比', a[6]
         ],
     '第一大类中六个子类的总体指标':
        ['预测销量与补偿后销量均比真实销量低，补偿变好且发生在同侧，该种情况的单品数', a[7],
         '均低同侧变好占补偿变好单品数的百分比', a[8],
         '预测销量与补偿后销量均比真实销量高，补偿变好且发生在同侧，该种情况的单品数', a[9],
         '均高同侧变好占补偿变好单品数的百分比', a[10],
         '预测销量偏低，补偿后销量偏高，补偿变好且发生在对侧，该种情况的单品数', a[11],
         '预低补高对侧变好占补偿变好单品数的百分比', a[12],
         '预测销量偏高，补偿后销量偏低，补偿变好且发生在对侧，该种情况的单品数', a[13],
         '预高补低对侧变好占补偿变好单品数的百分比', a[14],
         '预测销量偏低，完全补偿，该种情况的单品数', a[15],
         '预低全补占补偿变好单品数的百分比', a[16],
         '预测销量偏高，完全补偿，该种情况的单品数', a[17],
         '预高全补占补偿变好单品数的百分比', a[18],
         '补偿变好且发生在同侧的单品数', a[19],
         '补好同侧单品数占补偿变好单品数的百分比', a[20],
         '补偿变好且发生在对侧的单品数', a[21],
         '补好对侧单品数占补偿变好单品数的百分比', a[22],
         '完全补偿的单品数', a[23],
         '完全补偿的单品数占补偿变好单品数的百分比', a[24]
         ],
     '1.1 预测补偿均低，补偿同侧变好时各项指标的均值':
        ['1.1中预测偏差量的平均值', a[25][1],
         '1.1中预测偏差量与真实销量百分比的平均值', a[26][1],
         '1.1中补偿后剩余偏差量的平均值', a[27][1],
         '1.1中补偿后剩余偏差量与真实销量百分比的平均值', a[28][1],
         '1.1中补偿量的平均值', a[29][1],
         '1.1中补偿量占预测偏差量百分比的平均值', a[30][1]
         ],
     '1.2 预测补偿均高，补偿同侧变好时各项指标的均值':
        ['1.2中预测偏差量的平均值', a[31][1],
         '1.2中预测偏差量与真实销量百分比的平均值', a[32][1],
         '1.2中补偿后剩余偏差量的平均值', a[33][1],
         '1.2中补偿后剩余偏差量与真实销量百分比的平均值', a[34][1],
         '1.2中补偿量的平均值', a[35][1],
         '1.2中补偿量占预测偏差量百分比的平均值', a[36][1]
         ],
     '1.3 预测偏低补偿偏高，补偿对侧变好时各项指标的均值':
        ['1.3中预测偏差量的平均值', a[37][1],
         '1.3中预测偏差量与真实销量百分比的平均值', a[38][1],
         '1.3中补偿后剩余偏差量的平均值', a[39][1],
         '1.3中补偿后剩余偏差量与真实销量百分比的平均值', a[40][1],
         '1.3中补偿量的平均值', a[41][1],
         '1.3中补偿量占预测偏差量百分比的平均值', a[42][1]
         ],
     '1.4 预测偏高补偿偏低，补偿对侧变好时各项指标的均值':
        ['1.4中预测偏差量的平均值', a[43][1],
         '1.4中预测偏差量与真实销量百分比的平均值', a[44][1],
         '1.4中补偿后剩余偏差量的平均值', a[45][1],
         '1.4中补偿后剩余偏差量与真实销量百分比的平均值', a[46][1],
         '1.4中补偿量的平均值', a[47][1],
         '1.4中补偿量占预测偏差量百分比的平均值', a[48][1]
         ],
     '1.5 预测偏低，完全补偿时各项指标的均值':
        ['1.5中预测偏差量的平均值', a[49][1],
         '1.5中预测偏差量与真实销量百分比的平均值', a[50][1],
         '1.5中补偿后剩余偏差量的平均值', a[51][1],
         '1.5中补偿后剩余偏差量与真实销量百分比的平均值', a[52][1],
         '1.5中补偿量的平均值', a[53][1],
         '1.5中补偿量占预测偏差量百分比的平均值', a[54][1]
         ],
     '1.6 预测偏高，完全补偿时各项指标的均值':
        ['1.6中预测偏差量的平均值', a[55][1],
         '1.6中预测偏差量与真实销量百分比的平均值', a[56][1],
         '1.6中补偿后剩余偏差量的平均值', a[57][1],
         '1.6中补偿后剩余偏差量与真实销量百分比的平均值', a[58][1],
         '1.6中补偿量的平均值', a[59][1],
         '1.6中补偿量占预测偏差量百分比的平均值', a[60][1]
         ],
     '第二大类中六个子类的总体指标':
        ['预测销量与补偿后销量均比真实销量低，补偿变差且发生在同侧，该种情况的单品数', a[61],
         '均低同侧变差占补偿变差单品数的百分比', a[62],
         '预测销量与补偿后销量均比真实销量高，补偿变差且发生在同侧，该种情况的单品数', a[63],
         '均高同侧变差占补偿变差单品数的百分比', a[64],
         '预测销量偏低，补偿后销量偏高，补偿变差且发生在对侧，该种情况的单品数', a[65],
         '预低补高对侧变差占补偿变差单品数的百分比', a[66],
         '预测销量偏高，补偿后销量偏低，补偿变差且发生在对侧，该种情况的单品数', a[67],
         '预高补低对侧变差占补偿变差单品数的百分比', a[68],
         '预测销量偏低，完全补偿，该种情况的单品数', a[69],
         '预低全补占补偿变差单品数的百分比', a[70],
         '预测销量偏高，完全补偿，该种情况的单品数', a[71],
         '预高全补占补偿变差单品数的百分比', a[72],
         '补偿变差且发生在同侧的单品数', a[73],
         '补好同侧单品数占补偿变差单品数的百分比', a[74],
         '补偿变差且发生在对侧的单品数', a[75],
         '补好对侧单品数占补偿变差单品数的百分比', a[76],
         '完全补偿的单品数', a[77],
         '完全补偿的单品数占补偿变差单品数的百分比', a[78]
         ],
     '2.1 预测补偿均低，补偿同侧变差时各项指标的均值':
        ['2.1中预测偏差量的平均值', a[79][1],
         '2.1中预测偏差量与真实销量百分比的平均值', a[80][1],
         '2.1中补偿后剩余偏差量的平均值', a[81][1],
         '2.1中补偿后剩余偏差量与真实销量百分比的平均值', a[82][1],
         '2.1中补偿量的平均值', a[83][1],
         '2.1中补偿量占预测偏差量百分比的平均值', a[84][1]
         ],
     '2.2 预测补偿均高，补偿同侧变差时各项指标的均值':
        ['2.2中预测偏差量的平均值', a[85][1],
         '2.2中预测偏差量与真实销量百分比的平均值', a[86][1],
         '2.2中补偿后剩余偏差量的平均值', a[87][1],
         '2.2中补偿后剩余偏差量与真实销量百分比的平均值', a[88][1],
         '2.2中补偿量的平均值', a[89][1],
         '2.2中补偿量占预测偏差量百分比的平均值', a[90][1]
         ],
     '2.3 预测偏低补偿偏高，补偿对侧变差时各项指标的均值':
        ['2.3中预测偏差量的平均值', a[91][1],
         '2.3中预测偏差量与真实销量百分比的平均值', a[92][1],
         '2.3中补偿后剩余偏差量的平均值', a[93][1],
         '2.3中补偿后剩余偏差量与真实销量百分比的平均值', a[94][1],
         '2.3中补偿量的平均值', a[95][1],
         '2.3中补偿量占预测偏差量百分比的平均值', a[96][1]
         ],
     '2.4 预测偏高补偿偏低，补偿对侧变差时各项指标的均值':
        ['2.4中预测偏差量的平均值', a[97][1],
         '2.4中预测偏差量与真实销量百分比的平均值', a[98][1],
         '2.4中补偿后剩余偏差量的平均值', a[99][1],
         '2.4中补偿后剩余偏差量与真实销量百分比的平均值', a[100][1],
         '2.4中补偿量的平均值', a[101][1],
         '2.4中补偿量占预测偏差量百分比的平均值', a[102][1]
         ],
     '2.5 完全准确预测，补偿偏低时各项指标的均值':
        ['2.5中预测偏差量的平均值', a[103][1],
         '2.5中预测偏差量与真实销量百分比的平均值', a[104][1],
         '2.5中补偿后剩余偏差量的平均值', a[105][1],
         '2.5中补偿后剩余偏差量与真实销量百分比的平均值', a[106][1],
         '2.5中补偿量的平均值', a[107][1],
         '2.5中预测偏差量占补偿量百分比的平均值', a[108][1]
         ],
     '2.6 完全准确预测，补偿偏高时各项指标的均值':
        ['2.6中预测偏差量的平均值', a[109][1],
         '2.6中预测偏差量与真实销量百分比的平均值', a[110][1],
         '2.6中补偿后剩余偏差量的平均值', a[111][1],
         '2.6中补偿后剩余偏差量与真实销量百分比的平均值', a[112][1],
         '2.6中补偿量的平均值', a[113][1],
         '2.6中预测偏差量占补偿量百分比的平均值', a[114][1]
         ]
     }

print(json.dumps(result2, indent=4, ensure_ascii=False))
