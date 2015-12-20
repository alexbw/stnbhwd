local t = require 'torch'
local grad = require 'autograd'
local image = require 'image'
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local optim = require 'optim'
local stn = require 'stn'
local hasQT,q = pcall(require, 'qt')

torch.manualSeed(0)

-- MNIST dataset 
---------------------------------
local batchSize = 256
local train, validation = paths.dofile("./distort_mnist.lua")(true, true, true, batchSize) -- batch, normalize, distort
local imageHeight, imageWidth = train.data:size(3), train.data:size(4)
local gridHeight = imageHeight
local gridWidth = imageWidth

-- Set up confusion matrix
---------------------------------
local confusionMatrix = optim.ConfusionMatrix({0,1,2,3,4,5,6,7,8,9})

-- Get the STN functions
---------------------------------
local gridGenerator = grad.functionalize(stn.AffineGridGeneratorBHWD(gridHeight, gridWidth)) -- grid is same size as image
local bilinearSampler = grad.functionalize(stn.BilinearSamplerBHWD())

-- Set up the localizer network
---------------------------------
locnet = nn.Sequential()
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
locnet:add(cudnn.SpatialConvolution(1,20,5,5))
locnet:add(cudnn.ReLU(true))
locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
locnet:add(cudnn.SpatialConvolution(20,20,5,5))
locnet:add(cudnn.ReLU(true))
locnet:add(nn.View(20*2*2))
locnet:add(nn.Linear(20*2*2,20))
locnet:add(cudnn.ReLU(true))
local outLayer = nn.Linear(20,6)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(6):fill(0)
bias[1]=1
bias[5]=1
outLayer.bias:copy(bias)
locnet:add(outLayer)
locnet:cuda()

-- Set up classifier network
---------------------------------
model = nn.Sequential()
model:add(nn.View(32*32))
model:add(nn.Linear(32*32, 128))
model:add(cudnn.ReLU(true))
model:add(nn.Linear(128, 128))
model:add(cudnn.ReLU(true))
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())
model:cuda()

-- Functionalize networks
---------------------------------
local agLocnet, locParams = grad.functionalize(locnet)
local agClassnet, classParams = grad.functionalize(model)
local criterion = grad.nn.ClassNLLCriterion()

-- Set up parameters
---------------------------------
params = {
   locParams = locParams,
   classParams = classParams,
}

-- Define our loss function
---------------------------------
local function f(inputs, bhwdImages, labels)
   -- Reshape the image for the convnet
   local input = torch.view(bhwdImages, 
                     batchSize, 1, imageHeight, imageWidth)

   -- Calculate how we should warp the image
   local warpPrediction = agLocnet(inputs.locParams, input)

   -- Get warped grids
   local grids = gridGenerator(torch.view(warpPrediction, batchSize, 2, 3))

   -- Resample the images
   local resampledImages = bilinearSampler({bhwdImages, grids})

   -- Run the classifier on the warped images
   local warpedInput = torch.view(resampledImages,
                     batchSize, 
                     gridHeight*gridWidth)

   -- Predict image class on warped image
   local prediction = agClassnet(inputs.classParams, warpedInput)

   -- Calculate loss
   local loss = criterion(prediction, labels)

   return loss, prediction, resampledImages
end

local g = grad(f, {optimize = true})

local state = {learningRate=1e-2}
local optimfn, states = grad.optim.adagrad(g, state, params)

-- local state = {learningRate = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8}
-- local optimfn, states = grad.optim.adam(g, state, params)

local w1, w2
local nBatches = train:getNumBatches()
for epoch=1,100 do
   print("Epoch " .. epoch)
   for i=1,nBatches do
      xlua.progress(i,nBatches)

      -- Get images in BHWD format, labels in one-hot format:
      local data, labels, n = train:getBatch(i)
      local bhwdImages = data:transpose(2,3):cuda():transpose(3,4):cuda()
      local target = labels:cuda()

      -- Calculate gradients:
      local grads, loss, prediction, resampledImages = optimfn(bhwdImages, target)

      -- Log performance:
      confusionMatrix:batchAdd(prediction, target)
      if i % 10 == 0 then
      -- if i > 0 then
      -- if i == 10 then
         print(confusionMatrix)
         confusionMatrix:zero()
         local transformedImage = resampledImages:select(4,1)
         if hasQT then
            local origImage = bhwdImages:select(4,1)
            w1=image.display({image=origImage, nrow=16, legend='Original', win=w1})
            w2=image.display({image=transformedImage, nrow=16, legend='Resampled', win=w2})
         end
      end
   end
end

