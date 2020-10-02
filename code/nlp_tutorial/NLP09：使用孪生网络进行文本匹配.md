# 1.孪生(Siamese)网络基本原理

孪生网络是包含两个或多个相同的的子网络组件的神经网络，如下所示：

<img src="https://raw.githubusercontent.com/chongzicbo/images/master/imgimage-20200929214609062.png" alt="image-20200929214609062" style="zoom:67%;" />

在孪生网络中，不仅子网络的架构是相同的，而且权重在子网络中也是共享的，这样的网络被称为孪生网络。孪生网络背后的思想是其能够学习有用的数据描述符，这些描述符可以进一步用于在各个子网的输入之间进行比较。因此，孪生网络的输入可以是数值数据、图像数据(CNN为子网络)或者序列数据（以RNN为子网络）。

通常，孪生网络对输出执行二分类，判断输入是不是属于同一类。最常用的损失函数为：
$$
L=-ylogp+(1-y)log(1-p)
$$
其中，$L$是损失函数，$y$是类别标签0,1，$p$是预测的概率值。为了训练网络以区分相似和不相似地对象，可以一次提供一个正例和一个负例，并对损失进行加和：
$$
L=L_++L\_
$$
另外，也可以使用triplet loss:
$$
L=max(d(a,p)-d(a,n)+m,0)
$$
其中，$d$是距离函数，如L2损失；$a$是数据集中的一个样本,$p$是一个随机正样本，$n$是一个负样本，$m$是阈值。通过最小化上述损失函数，a与p之间的距离d(a,p)=0，而a与n之间的距离d(a,n)大于d(a,p)+margin。当negative example很好识别时，上述损失函数为0，否则是一个比较大的值。

# 2.使用孪生网络对Mnist数据集进行分类

```python
import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
from tqdm import tqdm
```

```python
do_learn=True
save_frequency=2
batch_size=64
lr=0.001
num_epochs=10
weight_decay=0.0001
```

```python
def get_int(b):
  return int(codecs.encode(b,"hex"),16)

def read_label_file(path):
  with open(path,"rb")as f:
    data=f.read()
  assert get_int(data[:4])==2049
  length=get_int(data[4:8])
  parsed=np.frombuffer(data,dtype=np.uint8,offset=8)
  return torch.from_numpy(parsed).view(length).long()

def read_image_file(path):
  with open(path,"rb") as f:
    data=f.read()

  assert get_int(data[:4])==2051
  length=get_int(data[4:8])    
  num_rows=get_int(data[8:12])
  num_cols=get_int(data[12:16])
  images=[]
  parsed=np.frombuffer(data,dtype=np.uint8,offset=16)
  return torch.from_numpy(parsed).view(length,num_rows,num_cols)
```

```python
class BalancedMNISTPair(torch.utils.data.Dataset):
  """Dataset that on each iteration provides two random pairs of
  MNIST images. One pair is of the same number (positive sample), one
  is of two different numbers (negative sample).
  """
  urls = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
  ]
  raw_folder = 'raw'
  processed_folder = 'processed'
  training_file = 'training.pt'
  test_file = 'test.pt'
  
  def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train # training set or test set
    
    if download:
        self.download()
        
    if not self._check_exists():
        raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        
    if self.train:
        self.train_data, self.train_labels = torch.load(
          os.path.join(self.root, self.processed_folder, self.training_file))
        
        train_labels_class = []
        train_data_class = []
        for i in range(10):
          indices = torch.squeeze((self.train_labels == i).nonzero())
          train_labels_class.append(torch.index_select(self.train_labels, 0, indices))
          train_data_class.append(torch.index_select(self.train_data, 0, indices))
          
        # generate balanced pairs
        self.train_data = []
        self.train_labels = []
        lengths = [x.shape[0] for x in train_labels_class]
        for i in range(10):
          for j in range(500): # create 500 pairs
              rnd_cls = random.randint(0,8) # choose random class that is not the same class
              if rnd_cls >= i:
                rnd_cls = rnd_cls + 1

              rnd_dist = random.randint(0, 100)
                
              self.train_data.append(torch.stack([train_data_class[i][j], train_data_class[i][j+rnd_dist], train_data_class[rnd_cls][j]]))
              self.train_labels.append([1,0])

        self.train_data = torch.stack(self.train_data)
        self.train_labels = torch.tensor(self.train_labels)
              
    else:
        self.test_data, self.test_labels = torch.load(
          os.path.join(self.root, self.processed_folder, self.test_file))
        
        test_labels_class = []
        test_data_class = []
        for i in range(10):
          indices = torch.squeeze((self.test_labels == i).nonzero())
          test_labels_class.append(torch.index_select(self.test_labels, 0, indices))
          test_data_class.append(torch.index_select(self.test_data, 0, indices))
          
        # generate balanced pairs
        self.test_data = []
        self.test_labels = []
        lengths = [x.shape[0] for x in test_labels_class]
        for i in range(10):
          for j in range(500): # create 500 pairs
              rnd_cls = random.randint(0,8) # choose random class that is not the same class
              if rnd_cls >= i:
                rnd_cls = rnd_cls + 1

              rnd_dist = random.randint(0, 100)
                
              self.test_data.append(torch.stack([test_data_class[i][j], test_data_class[i][j+rnd_dist], test_data_class[rnd_cls][j]]))
              self.test_labels.append([1,0])

        self.test_data = torch.stack(self.test_data)
        self.test_labels = torch.tensor(self.test_labels)
        
  def __getitem__(self, index):
    if self.train:
        imgs, target = self.train_data[index], self.train_labels[index]
    else:
        imgs, target = self.test_data[index], self.test_labels[index]
        
    img_ar = []
    for i in range(len(imgs)):
        img = Image.fromarray(imgs[i].numpy(), mode='L')
        if self.transform is not None:
          img = self.transform(img)
        img_ar.append(img)
        
    if self.target_transform is not None:
        target = self.target_transform(target)
        
    return img_ar, target
  
  def __len__(self):
    if self.train:
        return len(self.train_data)
    else:
        return len(self.test_data)
    
  def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
        os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
  
  def download(self):
    """Download the MNIST data if it doesn't exist in processed_folder already."""
    from six.moves import urllib
    import gzip

    if self._check_exists():
        return

    # download files
    try:
        os.makedirs(os.path.join(self.root, self.raw_folder))
        os.makedirs(os.path.join(self.root, self.processed_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
          pass
        else:
          raise

    for url in self.urls:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        with open(file_path, 'wb') as f:
          f.write(data.read())
        with open(file_path.replace('.gz', ''), 'wb') as out_f, \
              gzip.GzipFile(file_path) as zip_f:
          out_f.write(zip_f.read())
        os.unlink(file_path)

    # process and save as torch files
    print('Processing...')

    training_set = (
        read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
    )
    with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
        torch.save(test_set, f)

    print('Done!')

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    tmp = 'train' if self.train is True else 'test'
    fmt_str += '    Split: {}\n'.format(tmp)
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
```

```python
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(1,64,7)
    self.pool1=nn.MaxPool2d(2)
    self.conv2=nn.Conv2d(64,128,5)
    self.conv3=nn.Conv2d(128,256,5)
    self.linear1=nn.Linear(2304,512)
    self.linear2=nn.Linear(512,2)

  def forward(self,data):
    res=[]
    for i in range(2):
      x=data[i]
      x=self.conv1(x)
      x=F.relu(x)
      x=self.pool1(x)
      x=self.conv2(x)
      x=F.relu(x)
      x=self.conv3(x)
      x=F.relu(x)

      x=x.view(x.shape[0],-1)
      x=self.linear1(x)
      res.append(F.relu(x))

    res=torch.abs(res[1]-res[0])
    res=self.linear2(res)
    return res
```

```python
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(1,64,7)
    self.pool1=nn.MaxPool2d(2)
    self.conv2=nn.Conv2d(64,128,5)
    self.conv3=nn.Conv2d(128,256,5)
    self.linear1=nn.Linear(2304,512)
    self.linear2=nn.Linear(512,2)

  def forward(self,data):
    res=[]
    for i in range(2):
      x=data[i]
      x=self.conv1(x)
      x=F.relu(x)
      x=self.pool1(x)
      x=self.conv2(x)
      x=F.relu(x)
      x=self.conv3(x)
      x=F.relu(x)

      x=x.view(x.shape[0],-1)
      x=self.linear1(x)
      res.append(F.relu(x))

    res=torch.abs(res[1]-res[0])
    res=self.linear2(res)
    return res
```



原文链接：https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18