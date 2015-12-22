local testFileName = 'mnist.t7/test_32x32-blur.t7'
local trainFileName = 'mnist.t7/train_32x32-blur.t7'

local train = torch.load(trainFileName, 'ascii')
local test = torch.load(testFileName, 'ascii')
train.data = train.data:float()
train.labels = train.labels:float()
test.data = test.data:float()
test.labels = test.labels:float()
