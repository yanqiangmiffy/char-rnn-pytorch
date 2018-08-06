import pickle
import torch
from utils import load_data,random_train_example,time_since
import torch.nn as nn
from model import RNN
import time
import matplotlib.pyplot as plt
# 加载数据
n_letters,n_categories,_,_,_=load_data()

# 参数设置
criterion = nn.NLLLoss()
learning_rate = 0.0005
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
rnn = RNN(n_categories,n_letters,128, n_letters)


# 相关计算
def compute(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)


# 训练
def train():
    total_loss = 0 # Reset every plot_every iters
    start = time.time()
    for iter in range(1, n_iters + 1):
        output, loss = compute(*random_train_example())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (time_since(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    model_dir='result/model.pkl'
    loss_dir = 'result/all_losses.pkl'
    print("已保存训练数据:",loss_dir,model_dir)

    with open(loss_dir, 'wb') as out_data:
        pickle.dump(all_losses,out_data,pickle.HIGHEST_PROTOCOL)
    torch.save(rnn,model_dir)

if __name__ == '__main__':
    train()


    loss_dir = 'result/all_losses.pkl'
    with open(loss_dir, 'rb') as in_data:
        all_losses = pickle.load(in_data)

    plt.figure()
    plt.plot(all_losses)
    plt.show()

#CUDA_VISIBLE_DEVICES=1 python train.py