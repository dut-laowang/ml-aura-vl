import os
import shutil

# 源文件夹（imgs）
root_dir = "/workspace/ml-aura-vl/MM-SafetyBench/data/imgs"

# 目标文件夹（集中存放重命名图像）
output_dir = "/workspace/ml-aura-vl/MM-SafetyBench/data/pics"
os.makedirs(output_dir, exist_ok=True)

# 场景列表（13类）
SCENARIOS = [
    "01-Illegal_Activitiy", "02-HateSpeech", "03-Malware_Generation", "04-Physical_Harm",
    "05-EconomicHarm", "06-Fraud", "07-Sex", "08-Political_Lobbying", "09-Privacy_Violence",
    "10-Legal_Opinion", "11-Financial_Advice", "12-Health_Consultation", "13-Gov_Decision"
]

for scenario in SCENARIOS:
    sd_dir = os.path.join(root_dir, scenario, "SD")
    if not os.path.exists(sd_dir):
        print(f"⚠️ 跳过缺失路径: {sd_dir}")
        continue

    files = sorted(os.listdir(sd_dir), key=lambda x: int(os.path.splitext(x)[0]) if x.endswith(".jpg") else -1)
    copied_count = 0

    for fname in files:
        if not fname.endswith(".jpg"):
            continue
        src_path = os.path.join(sd_dir, fname)
        num = os.path.splitext(fname)[0]  # 原始编号，如 '10'
        dst_name = f"{scenario}_{num}.jpg"
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy(src_path, dst_path)
        copied_count += 1

    print(f"✅ 处理完成：{scenario}，共复制 {copied_count} 张图像")

print("🎉 所有 SD 图像已复制并重命名完毕！")
