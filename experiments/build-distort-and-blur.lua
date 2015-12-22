local torch = require 'torch'
local g = require 'autograd'
local stn = g.functionalize('stn')
local image = require 'image'

local function distortImage(thisImage, scale, rotation, translationX, translationY)
   scale = scale or 1
   rotation = rotation or 0
   translationX = translationX or 0
   translationY = translationY or 0

   local imageDepth, imageHeight, imageWidth
   if thisImage:nDimension() == 2 then
      imageHeight, imageWidth = thisImage:size(1), thisImage:size(2)
   elseif thisImage:nDimension() == 3 then
      imageDepth, imageHeight, imageWidth = thisImage:size(1), thisImage:size(2), thisImage:size(3)
   else
      error("Unsupported dimension number " .. thisImage:nDimension())
   end

   local gridGenerator = stn.AffineGridGeneratorBHWD(imageHeight, imageWidth)
   local matrixGenerator = stn.AffineTransformMatrixGenerator(true,true,true)
   local resampler = stn.BilinearSamplerBHWD()

   local params = torch.FloatTensor{{rotation, scale, translationX, translationY}}
   local A = matrixGenerator(params)
   local I = torch.transpose(torch.view(thisImage, imageDepth or 1, imageHeight, imageWidth, 1), 1, 4):contiguous()
   local grid = gridGenerator(A)
   local out = resampler({I, gridGenerator(A)})

   if imageDepth then
      return torch.transpose(torch.transpose(torch.squeeze(out, 1),2,3),1,2), torch.squeeze(A)
   else
      return torch.squeeze(torch.squeeze(out, 1), 3), torch.squeeze(A)
   end
end

local function distortData(images)
   local stn = g.functionalize('stn')
   local nImages = images:size(1)
   local transforms = torch.zeros(nImages, 2, 3):float()
   for i=1,nImages do
      if i % 1000 == 0 then
         print(i)
      end
      -- Sample rotation, translation, scale
      local thisImage = images:select(1,i)
      local scale = torch.uniform(0.7, 1.2)
      local rotation = torch.uniform(-3.14/4, 3.14/4)
      local translationX = torch.uniform(-0.5, 0.5)
      local translationY = torch.uniform(-0.5, 0.5)
      local I, A = distortImage(thisImage, scale, rotation, translationX, translationY)
      transforms[i] = A
      thisImage:copy(I)
   end
   return images, transforms
end

local function blurData(images, kernelSizes)
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

local testFileName = 'mnist.t7/test_32x32.t7'
local trainFileName = 'mnist.t7/train_32x32.t7'

local train = torch.load(trainFileName, 'ascii')
local test = torch.load(testFileName, 'ascii')
train.data = train.data:float()
train.labels = train.labels:float()
test.data = test.data:float()
test.labels = test.labels:float()

torch.manualSeed(0)

local kernelSizes = {1,8,16,32}
train.data, train.transform = blurData(distortData(train.data), kernelSizes)
test.data, test.transform = blurData(distortData(test.data), kernelSizes)

torch.save('mnist.t7/test_32x32-distort-and-blur.t7', test, 'binary')
torch.save('mnist.t7/train_32x32-distort-and-blur.t7', train, 'binary')
