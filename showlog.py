import matplotlib.pyplot as plt
import re

# 假設你的資料儲存在一個名為'log.txt'的檔案中
log_file = 'log.txt'

# 讀取檔案內容
with open(log_file, 'r', encoding='utf-8') as file:
    log_data = file.readlines()

# 用正則表達式提取每個epoch的loss值
epoch_losses = []
for line in log_data:
    match = re.search(r'train_loss=(\d+)', line)
    if match:
        epoch_losses.append(int(match.group(1)))

# 繪製圖表
# plt.plot(epoch_losses, marker='-o')
plt.plot(epoch_losses)
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()