local t = require 'torch'
local grad = require 'autograd'
local image = require 'image'
local nn = require 'nn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local optim = require 'optim'
local stn = require 'stn'

-- MNIST dataset 
---------------------------------
local batchSize = 256
local train, validation = paths.dofile("./distort_mnist.lua")(true, true, true, batchSize) -- batch, normalize, distort
local imageHeight, imageWidth = train.data:size(3), train.data:size(4)

-- Set up confusion matrix
---------------------------------
local confusionMatrix = optim.ConfusionMatrix({0,1,2,3,4,5,6,7,8,9})

-- Get the STN functions
---------------------------------
local fns = grad.functionalize('stn')
local gridHeight = imageHeight
local gridWidth = imageWidth
local matrixGenerator = fns.AffineTransformMatrixGenerator(true, true, true) -- rotation, scale, translation
local gridGenerator = fns.AffineGridGeneratorBHWD(gridHeight, gridWidth) -- grid is same size as image
local bilinearSampler = fns.BilinearSamplerBHWD()
-- local pooling = grad.functionalize('cudnn').SpatialMaxPooling(2,2,2,2)
-- print(pooling)

-- Set up our models
---------------------------------
-- Network that predicts warping from an image (conv then linear)
local nnetLocConv, nnetLocParamsConv = grad.model.SpatialNetwork({
   -- number of input features (maps):
   inputFeatures = 1, -- grayscale input
   hiddenFeatures = {20,20},
   poolings = {4,4},
   activations = 'ReLU',
   kernelSize = 3,
   -- batchNormalization = true,
})
x,y,n = train:getBatch(1)
out = nnetLocConv(nnetLocParamsConv, x:double())
nOut = torch.numel(out)/batchSize

local nnetLocLin, nnetLocParamsLin = grad.model.NeuralNetwork({
   inputFeatures = nOut,
   hiddenFeatures = {20,6},
   classifier = true,
   -- batchNormalization = true,
})

-- Network that predicts class from a warped image (just linear, for now)
local nnet, nnetParams = grad.model.NeuralNetwork({
   inputFeatures = gridHeight*gridWidth,
   hiddenFeatures = {128,128,10},
   activations = 'ReLU',
   classifier = true,
   -- batchNormalization = true,
})

-- Initialize the weights (really need a utility function for this!)
---------------------------------
for _,_params in pairs({nnetParams,nnetLocParamsConv,nnetLocParamsLin}) do
   for i,weights in ipairs(_params) do
      weights.W = weights.W:cuda()
      weights.b = weights.b:cuda()
      n = torch.numel(weights.b)
      weights.W:normal(0,0.01)
      weights.b:fill(0)
   end
end

-- Make sure that the net defaults to outputting identity transform
nnetLocParamsLin[#nnetLocParamsLin].W:fill(0)
local affineBias = nnetLocParamsLin[#nnetLocParamsLin].b:fill(0)
affineBias[1] = 1 -- identity transform
affineBias[5] = 1 
-- affineBias[1] = 0 -- rotation
-- affineBias[2] = 1 -- scale
-- affineBias[3] = 0 -- translationX
-- affineBias[4] = 0 -- translationy

-- Define our loss function
---------------------------------
local function f(inputs, bhwdImages, labels)
   -- Reshape the image for the convnet
   local input = torch.view(bhwdImages, 
                     batchSize, 1, imageHeight, imageWidth)

   -- Calculate how we should warp the image
   local convOut = nnetLocConv(inputs.nnetLocParamsConv, input)
   local warpPrediction = nnetLocLin(inputs.nnetLocParamsLin, torch.view(convOut, batchSize, nOut))

   -- Get 2D affine matrices
   -- local affineTransforms = matrixGenerator(warpPrediction)
   local affineTransforms = torch.view(warpPrediction, batchSize, 2, 3)

   -- Get warped grids
   local grids = gridGenerator(affineTransforms)

   -- Resample the images
   local resampledImages = bilinearSampler({bhwdImages, grids})

   -- Run the classifier on those input images
   local warpedInput = torch.view(resampledImages,
                     batchSize, 
                     gridHeight*gridWidth)

   -- Predict image class on warped image
   local out = nnet(inputs.nnetParams, warpedInput)
   local prediction = grad.util.logSoftMax(out)

   -- Calculate loss
   local loss = grad.loss.logMultinomialLoss(prediction, labels)
   return loss, prediction, resampledImages
end


-- Set up the params we'll feed into our function
params = {
   nnetParams=nnetParams, 
   nnetLocParamsLin=nnetLocParamsLin,
   nnetLocParamsConv=nnetLocParamsConv
}

-- Take the gradient of the function
local g = grad(f, {optimize = false})

-- state for sgd
local state = {learningRate = 1e-2, momentum = 0.9, weightDecay = 5e-4} -- state for SGD
local optimfn, states = grad.optim.sgd(g, state, params)

-- state for adam
-- local state = {learningRate = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8}
-- local optimfn, states = grad.optim.adam(g, state, params)

-- state for adagrad
-- local state = {learningRate=1e-2}
-- local optimfn, states = grad.optim.adagrad(g, state, params)

local w1, w2
local nBatches = train:getNumBatches()
for epoch=1,100 do
   print("Epoch " .. epoch)
   for i=1,nBatches do
      xlua.progress(i,nBatches)

      -- Get images in BHWD format, labels in one-hot format:
      local data, labels, n = train:getBatch(i)
      local bhwdImages = data:transpose(2,3):cuda():transpose(3,4):cuda()
      local target = grad.util.oneHot(labels, 10):cuda()

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
         -- local origImage = bhwdImages:select(4,1)
         -- w1=image.display({image=origImage, nrow=16, legend='Original', win=w1})
         w2=image.display({image=transformedImage, nrow=16, legend='Resampled', win=w2})
      end
   end
end
