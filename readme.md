# MNIST Pytorch Lightning
확장성을 고려하여 아래와 같이 분리하여 구현
```
.
├── dataset.py
├── loss.py
├── main.py
├── model.py
└── readme.md
```
* main.py 
  * 모델학습을 위한 파일
  * 아래와 같이 학습 실행
  * wandb logger로 실
```angular2html
python main.py --loss [loss] --lr [learning rate]
```

* model.py : 모델이 구현되어험있는 파일
* loss.py : loss 함수를 불러오는 파일
  * CE, MSE 가능
* dataset.py
  * Dataloader 구현

### 실험 결과
> Epoch : 10  
> loss : Cross Entropy  
> lr : 0.01  

* train_accuracy : 0.96875  
* train_loss : 0.1920  
* val_accuracy : 0.9369  
* val_loss : 0.2139  

[wandb 그래프](https://wandb.ai/kbum0617/mnist_test/runs/45trndpr/overview?workspace=user-kbum0617)