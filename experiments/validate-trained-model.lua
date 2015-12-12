-- Use rotation, translation and scaling for optimization


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
   opt.distortValidation = opt.distort
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
local train, validation = paths.dofile("../demo/distort_mnist.lua")(true, true, opt.distortValidation, opt.batchSize) -- batch, normalize, distort
local imageHeight, imageWidth = train.data:size(3), train.data:size(4)


-- Set up confusion matrix
---------------------------------
local confusionMatrix = optim.ConfusionMatrix({0,1,2,3,4,5,6,7,8,9})


-- Initialize the model
---------------------------------
local f
if opt.model == "stn" then
   -- Don't load params, we've already trained them
   f, _ = paths.dofile('stn-model.lua')(opt.batchSize, imageWidth, imageHeight)
elseif opt.model == "mlp" then
   -- Don't load params, we've already trained them
   f, _ = paths.dofile('mlp-model.lua')(opt.batchSize, imageWidth, imageHeight)
else
   print("Unrecognized model " .. opt.model)
end


-- Build an optimizer which warps an image
-- to increase classifier confidence
---------------------------------
local gridGenerator = grad.functionalize(stn.AffineGridGeneratorBHWD(imageHeight, imageWidth))
local bilinearSampler = grad.functionalize(stn.BilinearSamplerBHWD())
local criterion = grad.nn.ClassNLLCriterion()

local affineMatrices = torch.zeros(opt.batchSize, 2, 3):cuda()
function setIdentity(affineMatrices)
   affineMatrices:zero()
   for i=1,torch.size(affineMatrices,1) do
      affineMatrices[i][1][1] = 1.0
      affineMatrices[i][2][2] = 1.0
   end
   return affineMatrices
end
affineMatrices = setIdentity(affineMatrices)
local function warp(affineMatrices, params, bhwdImages, target)
   local grids = gridGenerator(affineMatrices)
   local resampledImages = bilinearSampler({bhwdImages, grids})
   local loss, prediction = f(params, resampledImages, target)
   local normPrediction = torch.cdiv(prediction, torch.expandAs(torch.sum(prediction,2), prediction)) + 1e-12
   local entropy = -torch.sum(torch.cmul(normPrediction,torch.log(normPrediction)))
   return entropy, resampledImages, prediction, normPrediction
   -- return -torch.sum(prediction), resampledImages, prediction
end
dwarp = grad(warp, {optimize=true})
local state = {learningRate=1e-3, momentum=0.9}

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
      affineMatrices = setIdentity(affineMatrices)
      local optimfn, states = grad.optim.sgd(dwarp, state, affineMatrices)
      for _=1,opt.warpIter do
         -- grads, loss, resampledImages, prediction = dwarp(affineMatrices, params, bhwdImages, target)
         grads, entropy, resampledImages, prediction = optimfn(params, bhwdImages, target)
         print("==============================")
         print(loss)
         print(target[1])
         pprint(torch.select(prediction,1,1))
         print(torch.select(affineMatrices,1,1))
         print("==============================")
         local transformedImage = resampledImages:select(4,1)
         w2=image.display({image=transformedImage, nrow=16, legend='Resampled', win=w2})
      end
      loss, prediction = f(params, resampledImages, target)

   else
      loss, prediction, resampledImages = f(params, bhwdImages, target)
         print("==============================")
         local entropy = -torch.sum(torch.cmul(prediction,torch.log(prediction)))
         local min, max = torch.min(prediction), torch.max(prediction)
         local eps = 1e-6
         local normPrediction = (prediction + min)/(max-min) + eps
         local entropy = -torch.sum(torch.cmul(normPrediction,torch.log(normPrediction)))
         print(torch.select(target,1,1))
         print("==============================")

   end

   -- Calculate loss and predictions:

   -- Log performance:
   confusionMatrix:batchAdd(prediction, target)
   if true then -- false then -- i % 50 == 0 then
      if hasQT and opt.display and resampledImages then
         local transformedImage = resampledImages:select(4,1)
         local origImage = bhwdImages:select(4,1)
         -- w1=image.display({image=origImage, nrow=16, legend='Original', win=w1})
         -- w2=image.display({image=transformedImage, nrow=16, legend='Resampled', win=w2})
         sys.sleep(1)
      end
   end
end

print(confusionMatrix)