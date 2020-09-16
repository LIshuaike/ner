import os
import time
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
from metric import F1Metric


class Model():
    def __init__(self, net, vocab, optimizer):
        self.net = net
        self.vocab = vocab
        self.optimizer = optimizer

    def fit(self, train_loader, dev_loader, test_loader, start_epoch, epochs,
            max_e, max_acc, interval, file):
        # 记录迭代时间
        total_time = 0

        for epoch in range(start_epoch, epochs):
            # 更新参数
            print(f"Epoch:{epoch}/{epochs}:")
            start = time.time()
            pbar = tqdm(enumerate(BackgroundGenerator(train_loader)),
                        total=len(train_loader))
            start_time = time.time()

            for i, data in pbar:
                words, chars, tags = data
                prepare_time = start_time - time.time()
                self.loss_batch(words, chars, tags)

                process_time = start_time - time.time() - prepare_time
                pbar.set_description(
                    f"training compute efficiency: {process_time / (process_time + prepare_time):.2f}"
                )
                start_time = time.time()

            # loss, metric_train = self.evaluate(train_loader)
            # print(
            #     f"{'train:':<6} Loss: {loss:.4f} Accuracy: {metric_train.score:.2%}"
            # )
            loss, metric_dev = self.evaluate(dev_loader)
            print(
                f"{'dev:':<6} Loss: {loss:.4f} Accuracy: {metric_dev.score:.2%}"
            )
            loss, metric_test = self.evaluate(test_loader)
            print(
                f"{'test:':<6} Loss: {loss:.4f} Accuracy: {metric_test.score:.2%}"
            )

            last = time.time() - start
            print(f"this epoch's time{last}\n")
            total_time += last

            # 保存效果最好的模型
            if metric_dev.score > max_acc:
                checkpoint = {
                    'max_e': max_e,
                    'max_acc': metric_dev.score,
                    'epoch': epoch + 1,
                    'net': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                ckpt_path = os.path.join(file, 'checkpoint.pth')
                torch.save(checkpoint, ckpt_path)
                max_e, max_acc = epoch, metric_dev.score
            elif epoch - max_e >= interval:
                break
        print(f"max accuracy of dev is {max_acc:.2%} at epoch {max_e}")
        print(f"mean time of each epoch is {total_time / epoch}s\n")

    def loss_batch(self, words, chars, tags):
        self.net.train()
        self.optimizer.zero_grad()
        mask = words.gt(0)
        output = self.net(words, chars)

        loss = self.net.crf(output, tags, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)  #梯度裁剪
        self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.net.eval()

        loss, right, total = 0, 0, 0
        metric = F1Metric(self.vocab)
        # 从加载器中加载数据进行评价
        pbar = tqdm(enumerate(BackgroundGenerator(dataloader)),
                    total=len(dataloader))
        start_time = time.time()
        for i, data in pbar:
            words, chars, tags = data
            prepare_time = start_time - time.time()

            mask = words.gt(0)
            lens = mask.sum(dim=1)
            target = torch.split(tags[mask], lens.tolist())
            output = self.net(words, chars)
            predict = self.net.crf.viterbi(output, mask)
            loss += self.net.crf(output, tags, mask)

            metric(predict, target)

            process_time = start_time - time.time() - prepare_time
            pbar.set_description(
                f"evaluating compute efficiency: {process_time / (process_time + prepare_time):.2f}"
            )
            start_time = time.time()

        loss /= len(dataloader)
        return loss, metric

    @torch.no_grad()
    def predict(self, dataloader):
        self.net.eval()

        loss, right, total = 0, 0, 0
        # 从加载器中加载数据进行评价
        with torch.no_grad():
            pbar = tqdm(enumerate(BackgroundGenerator(dataloader)),
                        total=len(dataloader))
            start_time = time.time()
            all_predicts, all_sentences = [], []
            for i, data in pbar:
                words, chars, tags = data
                prepare_time = start_time - time.time()

                output = self.net(words, chars)
                mask = words.gt(0)
                lens = mask.sum(dim=1)

                sentences = torch.split(words[mask], lens.tolist())
                predict = self.net.crf.viterbi(output, mask)

                sents = [self.vocab.id2word(seq) for seq in sentences]
                pred = [self.vocab.id2tag(seq) for seq in predict]

                all_predicts.extend(pred)
                all_sentences.extend(sents)

                process_time = start_time - time.time() - prepare_time
                pbar.set_description(
                    f"predicting compute efficiency: {process_time / (process_time + prepare_time):.2f}"
                )
                start_time = time.time()

        return all_sentences, all_predicts
