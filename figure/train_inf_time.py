import matplotlib.pyplot as plt

models = ['PALR', 'Proposed Model']
train_times = [4 + 50 / 60, 2 + 30 / 60]  # 4.833, 2.5 (hours)
inference_times = [47 + 20 / 60, 8]  # 47.333, 8 (hours)

# 서브플롯 생성 (1행 2열)
plt.figure(figsize=(12, 5))

# Training time subplot
plt.subplot(1, 2, 1)
bars = plt.bar(models, train_times, color=['skyblue', 'lightcoral'])
plt.xlabel('Models')
plt.ylabel('Time (hours)')
plt.ylim(0, max(train_times) + 5)
plt.title('Training Time', fontweight='bold')

# 화살표 추가 (첫 번째 막대 -> 두 번째 막대)
plt.arrow(0, train_times[0], 0.95, train_times[1] - train_times[0], 
          head_width=0.02, head_length=0.02, color='red', linewidth=2)
plt.text(0.5, (train_times[0] + train_times[1]) / 2, '-48%', 
         color='red', fontsize=12, fontweight='bold')

# Inference time subplot
plt.subplot(1, 2, 2)
bars = plt.bar(models, inference_times, color=['skyblue', 'lightcoral'])
plt.xlabel('Models')
plt.ylabel('Time (hours)')
plt.ylim(0, max(inference_times) + 5)
plt.title('Inference Time',fontweight='bold')

# 화살표 추가 (첫 번째 막대 -> 두 번째 막대)
plt.arrow(0, inference_times[0], 0.95, inference_times[1] - inference_times[0], 
          head_width=0.02, head_length=0.02, color='red', linewidth=2)
plt.text(0.5, (inference_times[0] + inference_times[1]) / 2, '-83%', 
         color='red', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
