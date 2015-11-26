-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz

function createDatasets(batch)
   local batch = batch or false
   local testFileName = 'mnist.t7/test_32x32.t7'
   local trainFileName = 'mnist.t7/train_32x32.t7'
   local train = torch.load(trainFileName, 'ascii')
   local test = torch.load(testFileName, 'ascii')
   train.data = train.data:float()
   train.labels = train.labels:float()
   test.data = test.data:float()
   test.labels = test.labels:float()
   
   local mean = train.data:mean()
   local std = train.data:std()
   train.data:add(-mean):div(std)
   test.data:add(-mean):div(std)
   
   local batchSize = 256
   
   if batch then
      local datasetTrain = {
         getBatch = function(self, idx)
            local data = train.data:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            local labels = train.labels:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            return data, labels, batchSize
         end,
         getNumBatches = function()
            return torch.floor(60000 / batchSize)
         end
      }
      
      local datasetVal = {
         getBatch = function(self, idx)
            local data = test.data:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            local labels = test.labels:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            return data, labels, batchSize
         end,
         getNumBatches = function()
            return torch.floor(10000 / batchSize)
         end
      }
      return datasetTrain, datasetVal
   else
      return train, test
   end
end

return createDatasets