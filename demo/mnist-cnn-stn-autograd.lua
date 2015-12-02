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
gridHeight = imageHeight/2
gridWidth = imageWidth/2
local gridGenerator = fns.AffineGridGeneratorBHWD(imageHeight/2, imageWidth/2) -- grid is same size as image
local bilinearSampler = fns.BilinearSamplerBHWD()


-- Set up our models
---------------------------------
-- Network that predicts warping from an image (conv then linear)
local nnetLocConv, nnetLocParamsConv = grad.model.SpatialNetwork({
   -- number of input features (maps):
   inputFeatures = 1, -- grayscale input
   hiddenFeatures = {20,20},
   poolings = {2,2},
   activations = 'Sigmoid',
   kernelSize = 5,
   -- dropoutProbs = {.1, .1},
})
x,y,n = train:getBatch(1)
out = nnetLocConv(nnetLocParamsConv, x:double())
nOut = torch.numel(out)/batchSize

local nnetLocLin, nnetLocParamsLin = grad.model.NeuralNetwork({
   inputFeatures = nOut,
   hiddenFeatures = {20,6},
   classifier = true,
})

-- Network that predicts class from a warped image (just linear, for now)
local nnet, nnetParams = grad.model.NeuralNetwork({
   inputFeatures = gridHeight*gridWidth,
   hiddenFeatures = {128,128,10},
   activations = 'ReLU',
   classifier = true,
})

print(nnetLocParamsConv)

-- Initialize the weights (really need a utility function for this!)
---------------------------------
for _,_params in pairs({nnetParams,nnetLocParamsConv,nnetLocParamsLin}) do
   for i,weights in ipairs(_params) do
      weights.W = weights.W:cuda()
      weights.b = weights.b:cuda()
      weights.W:normal(0,0.1)
      weights.b:fill(0)
   end
end

-- Make sure that the net defaults to outputting identity transform
local affineBias = nnetLocParamsLin[#nnetLocParamsLin].b:fill(0)
affineBias[1] = 1
affineBias[5] = 1
nnetLocParamsLin[#nnetLocParamsLin].W:fill(0)

-- Define our loss function
---------------------------------
local function f(inputs, bhwdImages, labels)
   -- Flatten the input for the MLP
   local input = torch.view(bhwdImages, 
                     batchSize, 1, imageHeight, imageWidth)

   -- Calculate how we should warp the image
   local convOut = nnetLocConv(inputs.nnetLocParamsConv, input)
   local warpPrediction = nnetLocLin(inputs.nnetLocParamsLin, torch.view(convOut, batchSize, nOut))

   -- Get 2D affine matrices
   local affineTransforms = torch.view(warpPrediction, batchSize, 2, 3)

   -- Get warped grids
   local grids = gridGenerator(affineTransforms)

   -- Resample the images
   local resampledImages = bilinearSampler({bhwdImages, grids})

   -- Run the classifier on those input images
   local warpedInput = torch.view(resampledImages,
                     batchSize, 
                     imageHeight*imageWidth)

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
local g = grad(f, {optimize = true})

local lr = 1e-2
local w1
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
      local grads, loss, prediction, resampledImages = g(params, bhwdImages, target)

      -- Update, making sure to scale update by batch size:
      for k=1,#params.nnetParams do
         params.nnetParams[k].W = params.nnetParams[k].W - grads.nnetParams[k].W*lr/n
         params.nnetParams[k].b = params.nnetParams[k].b - grads.nnetParams[k].b*lr/n
      end
      for k=1,#params.nnetLocParamsConv do
         params.nnetLocParamsConv[k].W = params.nnetLocParamsConv[k].W - grads.nnetLocParamsConv[k].W*lr/n/1e2
         params.nnetLocParamsConv[k].b = params.nnetLocParamsConv[k].b - grads.nnetLocParamsConv[k].b*lr/n/1e2
      end
      for k=1,#params.nnetLocParamsLin do
         params.nnetLocParamsLin[k].W = params.nnetLocParamsLin[k].W - grads.nnetLocParamsLin[k].W*lr/n/1e1
         params.nnetLocParamsLin[k].b = params.nnetLocParamsLin[k].b - grads.nnetLocParamsLin[k].b*lr/n/1e1
      end

      -- Log performance:
      confusionMatrix:batchAdd(prediction, target)
      if i % 10 == 0 then -- train:getNumBatches() then
      -- if i > 0 then
         print(confusionMatrix)
         confusionMatrix:zero()
         local showImage = resampledImages:select(4,1)
         w1=image.display({image=showImage, nrow=16, legend='Resampled', win=w1}) -- min=0, max=1, scaleeach=false, saturate=false, 
      end
   end
end
