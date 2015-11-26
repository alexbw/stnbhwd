local t = require 'torch'
grad = require 'autograd'
require 'nn'
require 'stn'

-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz

require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
paths.dofile('Optim.lua')

use_stn = true

-- MNIST dataset 
train, validation = require('./mnist.lua')(false)
train.data = train.data:double()
print(train.data:size())
validation.data = validation.data:double()
imgs = train.data:narrow(1,1,1)
-- show_img = image.scale(imgs:select(1,1),300,300,'simple')
local w1,w2,w3

-- Batch size of 2
-- Affine transformation matrix
M = torch.zeros(2,2,3)
for i=1,2 do
   M[i][1][1] = 1
   M[i][2][2] = 1
end

fns = grad.functionalize('stn')
gridGenerator = fns.AffineGridGeneratorBHWD(32,32)
matrixGenerator = fns.AffineTransformMatrixGenerator(true,true,true) -- rotation, scale, translation
bilinearSampler = fns.BilinearSamplerBHWD()

rotation = math.pi/4
scale = 1.
translation = {0,0}
transformParams = torch.Tensor{rotation,scale,translation[1],translation[2]}
transformParams = torch.view(transformParams,1,4)
transformMatrix = matrixGenerator(transformParams)

grid = gridGenerator(transformMatrix)
bhwdImages = imgs:transpose(2,3)
bhwdImages = bhwdImages:transpose(3,4)
-- print(imgs:size())
-- print(bhwdImages:size())
-- os.exit()
-- BHWD
-- BHW
resampledImage = bilinearSampler({bhwdImages,grid})
showImage = resampledImage:select(4,1)

w1=image.display({image=showImage, nrow=16, legend='Just trying to do this', win=w1})
-- KxHxW

-- print(transformMatrix)
-- print(gridGenerator(transformMatrix))
-- print(resampledImage)

-- print(fns.AffineGridGeneratorBHWD(32,32)(M))
-- f,params = grad.functionalize(a)(M)
-- print(f(M)) -- fails
-- print(f(params, M)) -- fails

-- print(a(M))
-- print(a)
-- gridMaker,params = grad.functionalize(nn.AffineGridGeneratorBHWD(32,32))
-- gridMaker({}, M:cuda())
-- print(what())
-- modelf, params = grad.functionalize(nn.AffineGridGeneratorBHWD(32,32))
-- print(what.AffinegridGeneratorBHWD)
-- print(params)

-- -- for epoch=1,30 do
-- --    model:training()
-- --    local trainError = 0
-- --    for batchidx = 1, datasetTrain:getNumBatches() do
-- --       local inputs, labels = datasetTrain:getBatch(batchidx)
-- --       err = optimizer:optimize(optim.sgd, inputs:cuda(), labels:cuda(), criterion)
-- --       --print('epoch : ', epoch, 'batch : ', batchidx, 'train error : ', err)
-- --       trainError = trainError + err
-- --    end
-- --    print('epoch : ', epoch, 'trainError : ', trainError / datasetTrain:getNumBatches())
   
-- --    model:evaluate()
-- --    local valError = 0
-- --    local correct = 0
-- --    local all = 0
-- --    for batchidx = 1, datasetVal:getNumBatches() do
-- --       local inputs, labels = datasetVal:getBatch(batchidx)
-- --       local pred = model:forward(inputs:cuda())
-- --       valError = valError + criterion:forward(pred, labels:cuda())
-- --       _, preds = pred:max(2)
-- --       correct = correct + preds:eq(labels:cuda()):sum()
-- --       all = all + preds:size(1)
-- --    end
-- --    print('validation error : ', valError / datasetVal:getNumBatches())
-- --    print('accuracy % : ', correct / all * 100)
-- --    print('')
   
-- --    if use_stn then
-- --       w1=image.display({image=spanet.output, nrow=16, legend='STN-transformed inputs, epoch : '..epoch, win=w1})
-- --       w2=image.display({image=tranet:get(1).output, nrow=16, legend='Inputs, epoch : '..epoch, win=w2})
-- --    end
   
-- -- end

