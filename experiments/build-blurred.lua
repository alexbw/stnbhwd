local torch = require 'torch'
local g = require 'autograd'
local stn = g.functionalize('stn')
local image = require 'image'

local testFileName = 'mnist.t7/test_32x32.t7'
local trainFileName = 'mnist.t7/train_32x32.t7'

local train = torch.load(trainFileName, 'ascii')
local test = torch.load(testFileName, 'ascii')
train.data = train.data:float()
train.labels = train.labels:float()
test.data = test.data:float()
test.labels = test.labels:float()

function blurData(images, kernelSizes)
   kernelSizes = kernelSizes or {1,2,4,8,16,32}
   nImages, imageHeight, imageWidth = images:size(1), images:size(3), images:size(4)
   local blurredImages = torch.zeros(nImages, #kernelSizes, imageHeight, imageWidth)
   local i = 1
   for i=1,nImages do
      if i % 1000 == 0 then
         print(i)
      end
      local I = images:select(1,i)
      for ik,k in pairs(kernelSizes) do
         blurredImages:select(1,i):select(1,ik):copy(image.convolve(I, image.gaussian(k), 'same'))
      end
   end
   return blurredImages
end


local kernelSizes = {1,2,4,8,16,32}
train.data = blurData(train.data, kernelSizes)
test.data = blurData(test.data, kernelSizes)

torch.save('mnist.t7/test_32x32-blur.t7', test, 'binary')
torch.save('mnist.t7/train_32x32-blur.t7', train, 'binary')
