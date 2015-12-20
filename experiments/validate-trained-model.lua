-- Use rotation, translation and scaling for optimization

-- REQUIRES NNFUNC-UPDATE BRANCH OF AUTOGRAD
-- qlua validate-trained-model.lua --nEpoch=100  --model=cnn --distort=false  --display=true --distortValidation=true --validationBatchSize=256 --warp=true --warpIter=50

require 'trepl'
local t = require 'torch'
local grad = require 'autograd'
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local optim = require 'optim'
local image = require 'image'
local stn = require 'stn'
local hasQT,q = pcall(require, 'qt')
local pprint = require 'pprint'

-- Options
local opt = lapp [[
Run benchmarks.

Options:
   --distort (default false)
   --model (default stn)
   --optimizer (default adagrad)
   --display (default false)
   --batchSize (default 256)
   --nEpoch (default 100)
   --distortValidation (default same) Validation set will be distorted like training, unless set differently
   --warp (default false)
   --warpIter (default 10)
   --validationBatchSize (default same)
]]

if opt.distort == "true" then opt.distort = true else opt.distort = false end
if opt.display == "true" then opt.display = true else opt.display = false end
if opt.warp == "true" then opt.warp = true else opt.warp = false end
if opt.distortValidation == "same" then 
   opt.distortValidation = opt.distort
elseif opt.distortValidation == "true" then
   opt.distortValidation = true
elseif opt.distortValidation == "false" then
   opt.distortValidation = false
else
   opt.validationBatchSize = opt.distort
end   
if opt.validationBatchSize == "same" then 
   opt.validationBatchSize = opt.batchSize
else
   opt.validationBatchSize = tonumber(opt.validationBatchSize)
end


torch.manualSeed(0)

-- Name of file to serialize once fitting is completed
outFile = string.format("model=%s-distort=%s-optimizer=%s-nepoch=%d-batchSize=%d.t7", 
   opt.model, 
   tostring(opt.distort), 
   opt.optimizer, 
   opt.nEpoch,
   opt.batchSize)

if not paths.filep(outFile) then
   print(opt)
   error("The model correponding to the passed opts does not exist: " .. outFile)
end

local params = t.load(outFile).params -- don't load the function, lacks upvalues


-- MNIST dataset 
---------------------------------
local train, validation = paths.dofile("../demo/distort_mnist.lua")(true, true, opt.distortValidation, opt.validationBatchSize) -- batch, normalize, distort
local imageHeight, imageWidth = train.data:size(3), train.data:size(4)


-- Set up confusion matrix
---------------------------------
local confusionMatrix = optim.ConfusionMatrix({0,1,2,3,4,5,6,7,8,9})


-- Initialize the model
---------------------------------
local f
if opt.model == "stn" then
   f, _ = paths.dofile('stn-model.lua')(opt.validationBatchSize, imageWidth, imageHeight)
elseif opt.model == "mlp" then
   f, _ = paths.dofile('mlp-model.lua')(opt.validationBatchSize, imageWidth, imageHeight)
elseif opt.model == "cnn" then
   f, _ = paths.dofile('cnn-model.lua')(opt.validationBatchSize, imageWidth, imageHeight)
else
   print("Unrecognized model " .. opt.model)
end


-- Build an optimizer which warps an image
-- to increase classifier confidence
---------------------------------
local matrixGenerator = grad.functionalize(stn.AffineTransformMatrixGenerator(true,true,true)) -- rotation, scale, translation
local gridGenerator = grad.functionalize(stn.AffineGridGeneratorBHWD(imageHeight, imageWidth))
local bilinearSampler = grad.functionalize(stn.BilinearSamplerBHWD())
local criterion = grad.nn.ClassNLLCriterion()

local affineMatrices = torch.zeros(opt.validationBatchSize, 2, 3):cuda()
local transformParams = torch.zeros(opt.validationBatchSize, 4):cuda()
function setIdentity(affineMatrices)
   affineMatrices:zero()
   for i=1,torch.size(affineMatrices,1) do
      affineMatrices[i][1][1] = 1.0
      affineMatrices[i][2][2] = 1.0
   end
   return affineMatrices
end
function setIdentityTransform(transformParams)
   transformParams:select(2,1):fill(0)
   transformParams:select(2,2):fill(1)
   transformParams:select(2,3):fill(0)
   transformParams:select(2,4):fill(0)
   return transformParams
end

affineMatrices = setIdentity(affineMatrices)
transformParams = setIdentityTransform(transformParams)

local function warpMatrix(affineMatrices, params, bhwdImages, target)
   local grids = gridGenerator(affineMatrices)
   local resampledImages = bilinearSampler({bhwdImages, grids})
   local loss, prediction = f(params, resampledImages, target)
   local normPrediction = torch.cdiv(prediction, torch.expandAs(torch.sum(prediction,2), prediction)) + 1e-12
   local entropy = -torch.sum(torch.cmul(normPrediction,torch.log(normPrediction)))
   return entropy, resampledImages, prediction, normPrediction
end
local function warp(transformParams, params, bhwdImages, target)
   local affineMatrices = matrixGenerator(transformParams)
   local grids = gridGenerator(affineMatrices)
   local resampledImages = bilinearSampler({bhwdImages, grids})
   print(torch.size(resampledImages))
   local loss, prediction = f(params, resampledImages, target)
   local entropy = torch.sum(-torch.sum(torch.cmul(prediction,torch.exp(prediction)),2))
   return entropy, torch.exp(prediction), resampledImages
end

dwarp = grad(warp, {optimize=false})
-- local state = {learningRate=5e-2}
-- local state = {learningRate=5e-2, momentum=0.5}
local state = {learningRate = 5e-2} -- , beta1 = 0.9, beta2 = 0.9, epsilon = 1e-8}

-- Run the validation
---------------------------------
local nBatches = validation:getNumBatches()
for i=1,nBatches do
   xlua.progress(i,nBatches)

   -- Get images in BHWD format, labels in one-hot format:
   local data, labels, n = validation:getBatch(i)
   local bhwdImages = data:transpose(2,3):cuda():transpose(3,4):cuda()
   local target = labels:cuda()

   -- Optionally warp
   local loss, prediction, resampledImages, grads, entropy
   if opt.warp then
      -- affineMatrices = setIdentity(affineMatrices)
      transformParams = setIdentityTransform(transformParams)
      -- local optimfn, states = grad.optim.adagrad(dwarp, state, transformParams)
      -- local optimfn, states = grad.optim.sgd(dwarp, state, transformParams)
      local optimfn, states = grad.optim.adam(dwarp, state, transformParams)
      for _=1,opt.warpIter do
         grads, entropy, prediction, resampledImages = optimfn(params, bhwdImages, target)
         local transformedImage = resampledImages:select(4,1)
         if hasQT and opt.display then
            w2=image.display({image=transformedImage, nrow=16, legend='Resampled', win=w2})
         end
      end
      loss, prediction = f(params, resampledImages, target)

   else
      loss, prediction, resampledImages = f(params, bhwdImages, target)
   end

   -- Calculate loss and predictions:

   -- Log performance:
   confusionMatrix:batchAdd(prediction, target)
   if true then -- false then -- i % 50 == 0 then
      if hasQT and opt.display and resampledImages then
         local transformedImage = resampledImages:select(4,1)
         local origImage = bhwdImages:select(4,1)
      end
   end
end

print(confusionMatrix)