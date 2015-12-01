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
local batchSize = 128
local train, validation = paths.dofile("./distort_mnist.lua")(true, true, false, batchSize) -- batch, normalize, distort
local imageHeight, imageWidth = train.data:size(3), train.data:size(4)

-- Set up confusion matrix
---------------------------------
local confusionMatrix = optim.ConfusionMatrix({0,1,2,3,4,5,6,7,8,9})

-- Get the STN functions
---------------------------------
local fns = grad.functionalize('stn')
local gridGenerator = fns.AffineGridGeneratorBHWD(imageHeight, imageWidth) -- grid is same size as image
local bilinearSampler = fns.BilinearSamplerBHWD()


-- Set up our models
---------------------------------
-- Network that predicts warping from an image
local nnetLoc, nnetLocParams = grad.model.NeuralNetwork({
   -- number of input features:
   inputFeatures = imageHeight*imageWidth,
   hiddenFeatures = {100,100,6},
   activations = 'ReLU',
   classifier = true,
})

-- Network that predicts class from a warped image
local nnet, nnetParams = grad.model.NeuralNetwork({
   inputFeatures = imageHeight*imageWidth,
   hiddenFeatures = {100,100,10},
   activations = 'ReLU',
   classifier = true,
})

-- Initialize the weights (really need a utility function for this!)
---------------------------------
for i,weights in ipairs(nnetParams) do
   weights.W = weights.W:cuda()
   weights.b = weights.b:cuda()
   n = weights.W:size(1)
   weights.W:uniform(-1/math.sqrt(n),1/math.sqrt(n))
   weights.b:fill(0)
end

for i,weights in ipairs(nnetLocParams) do
   weights.W = weights.W:cuda()
   weights.b = weights.b:cuda()
   n = weights.W:size(1)
   weights.W:uniform(-1/math.sqrt(n),1/math.sqrt(n))
   weights.b:fill(0)
end

-- Make sure that the net defaults to outputting identity transform
local affineBias = nnetLocParams[#nnetLocParams].b:fill(0)
affineBias[1] = 1
affineBias[5] = 1
nnetLocParams[#nnetLocParams].W:fill(0)

-- Define our loss function
---------------------------------
local function f(inputs, bhwdImages, labels)
   -- Flatten the input for the MLP
   local input = torch.view(bhwdImages, 
                     batchSize, 
                     imageHeight*imageWidth)

   -- Calculate how we should warp the image
   local warpPrediction = nnetLoc(inputs.nnetLocParams, input)

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
params = {nnetParams=nnetParams, nnetLocParams=nnetLocParams}

-- Take the gradient of the function
local g = grad(f, {optimize = true})

local lr = 1e-2
local w1
for epoch=1,100 do
   for i=1,train:getNumBatches() do

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
      for k=1,#params.nnetLocParams do
         params.nnetLocParams[k].W = params.nnetLocParams[k].W - grads.nnetLocParams[k].W*lr/n
         params.nnetLocParams[k].b = params.nnetLocParams[k].b - grads.nnetLocParams[k].b*lr/n
      end

      -- Log performance:
      confusionMatrix:batchAdd(prediction, target)
      if i == 1 then -- train:getNumBatches() then
      -- if i > 0 then
         print(confusionMatrix)
         confusionMatrix:zero()
         local showImage = resampledImages:select(4,1)
         w1=image.display({image=showImage, nrow=16, legend='Resampled', win=w1}) -- min=0, max=1, scaleeach=false, saturate=false, 
      end
   end
end
