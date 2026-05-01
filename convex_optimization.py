import numpy as np
from scipy.optimize import minimize

# ===================== 1. 业务参数（你直接替换成你的数据）=====================
# 5个客户档位：1档(最优) ~ 5档(最差)
n = np.array([1000, 1500, 2000, 1500, 1000])    # 每档客户数量
s = np.array([100, 80, 60, 40, 20])            # 每档平均年销售额（万）
mu = np.array([0.08, 0.06, 0.03, 0.01, -0.02])# 每档单位净收益率（利息-坏账）

# 固定收益系数 c_k = mu_k * n_k * s_k（线性收益项）
c = mu * n * s  

# 凸优化核心：二次惩罚系数（调大=更均衡，解更靠中间；调小=更激进）
gamma = 0.5  

# 业务约束
TOTAL_MIN = 5000    # 总贷款额度下限（万）
TOTAL_MAX = 20000   # 总贷款额度上限（万）
X_MAX = 0.8         # 单档授信比例上限（不超过销售额80%）
RISK_RATIO = 0.25   # 高风险档(4+5档)集中度上限25%

# ===================== 2. 凸优化目标函数 =====================
# 目标：最大化 总净收益 - gamma*平方和（凹函数 → 最优解在内部）
# scipy 只能最小化，所以取负数
def objective(x):
    # 总净收益（线性项）
    total_profit = np.sum(c * x)  
    # 二次惩罚项（防止解卡在边界，强制均衡）
    penalty = gamma * np.sum(x ** 2)  
    # 最小化 = 最大化 (收益-惩罚)
    return - (total_profit - penalty)  

# ===================== 3. 线性约束条件 =====================
constraints = []

# 约束1：总贷款规模约束 TOTAL_MIN ≤ sum(n*s*x) ≤ TOTAL_MAX
def total_amt(x):
    return np.sum(n * s * x) - TOTAL_MIN
constraints.append({'type': 'ineq', 'fun': total_amt})

def total_amt_max(x):
    return TOTAL_MAX - np.sum(n * s * x)
constraints.append({'type': 'ineq', 'fun': total_amt_max})

# 约束2：高风险档(4、5档) 余额占比 ≤ 25%
def risk_concentration(x):
    high_risk = np.sum(n[3:] * s[3:] * x[3:])
    total = np.sum(n * s * x)
    return RISK_RATIO * total - high_risk
constraints.append({'type': 'ineq', 'fun': risk_concentration})

# 约束3：授信比例 0 ≤ x_k ≤ X_MAX（变量边界）
bounds = [(0, X_MAX) for _ in range(5)]

# ===================== 4. 启动凸优化求解 =====================
x0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 初始值（随便设）
result = minimize(objective, x0, method='SLSQP', 
                  bounds=bounds, constraints=constraints)

# ===================== 5. 输出结果 =====================
print("="*60)
print("凸优化求解成功！最优授信比例（每档x_k）：")
for i in range(5):
    print(f"第{i+1}档: {result.x[i]:.4f}")
print("="*60)
print(f"总净收益: {np.sum(c*result.x):.2f} 万元")
print(f"总投放额度: {np.sum(n*s*result.x):.2f} 万元")
print(f"高风险档占比: {np.sum(n[3:]*s[3:]*result.x[3:])/np.sum(n*s*result.x):.2%}")