----------------------------------------------------------------------
-- Based on https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'paths'
mnist_cluttered = require 'mnist_cluttered'

cudnn.benchmark = true

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a CNN on Translated Cluttered MNIST')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-decayEpochs', 50, 'decrease learning rate by a factor of 10 every N epochs')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-weightDecay', 0, 'L2 regularization of weights')
cmd:option('-optimizer', "SGD", 'optimization method: SGD, ADAM')

cmd:option('-batchSize', 32, 'number of examples per batch')
cmd:option('-numEpochs', 100, 'number of epochs')
cmd:option('-coarseModel', false, 'use coarse model (default is fine model)')
cmd:option('-manualSeed', 2, 'Manually set RNG seed')

cmd:option('-GPU', 0, 'ID of GPU to use (zero-based!)')
cmd:option('-save', '.', 'folder to save results')

cmd:option('-fixedTransformations', false, 'do not randomize translations and cluttering, making effective dataset size smaller')

cmd:text()

opt = cmd:parse(arg or {})
print(opt)

cutorch.setDevice(opt.GPU + 1)
torch.manualSeed(opt.manualSeed)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {100, 100}

model = nn.Sequential()
if opt.coarseModel then
   print('Building coarse model (2 conv layers)')
   -- Coarse layers: 2 convolutional layers, with 7 × 7 and 3 × 3 filter sizes, 12 and 24 filters, respectively.
   -- We use a stride a of 2 × 2 for both layers.
   model:add(cudnn.SpatialConvolution(1, 12, 7, 7, 2, 2, 0, 0))
   model:add(nn.SpatialBatchNormalization(12))
   model:add(cudnn.ReLU(true))
   model:add(cudnn.SpatialConvolution(12, 24, 3, 3, 2, 2, 0, 0))
   model:add(nn.SpatialBatchNormalization(24))
   model:add(cudnn.ReLU(true))
else
   print('Building fine model (5 conv layers)')
   -- Fine layers: 5 convolutional layers, each with 3×3 filter sizes, 1×1 strides, and 24 filters.
   -- We apply 2 × 2 pooling with 2 × 2 stride after the second and fourth layers.
   -- We also use 1 × 1 zero padding in all layers except for the first and last layers.
   model:add(cudnn.SpatialConvolution(1, 24, 3, 3, 1, 1, 0, 0))
   model:add(nn.SpatialBatchNormalization(24))
   model:add(cudnn.ReLU(true))
   model:add(cudnn.SpatialConvolution(24, 24, 3, 3, 1, 1, 1, 1))
   model:add(nn.SpatialBatchNormalization(24))
   model:add(cudnn.ReLU(true))
   model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(cudnn.SpatialConvolution(24, 24, 3, 3, 1, 1, 1, 1))
   model:add(nn.SpatialBatchNormalization(24))
   model:add(cudnn.ReLU(true))
   model:add(cudnn.SpatialConvolution(24, 24, 3, 3, 1, 1, 1, 1))
   model:add(nn.SpatialBatchNormalization(24))
   model:add(cudnn.ReLU(true))
   model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(cudnn.SpatialConvolution(24, 24, 3, 3, 1, 1, 0, 0))
   model:add(nn.SpatialBatchNormalization(24))
   model:add(cudnn.ReLU(true))
end

-- Top layers: one convolutional layer with 4 × 4 filter size, 2 × 2 stride and 96 filters,
-- followed by global max pooling.
-- The result is fed into a softmax layer with 10 outputs corresponding to each digit.
model:add(cudnn.SpatialConvolution(24, 96, 4, 4, 2, 2, 0, 0))
model:add(nn.SpatialBatchNormalization(96))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(10, 10, 1, 1))
model:add(nn.View(96))
model:add(nn.Linear(96, #classes))
model:add(cudnn.LogSoftMax())

print('Model to train:')
print(model)

model = model:cuda()
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

-- retrieve parameters and gradients
parameters, gradParameters = model:getParameters()

----------------------------------------------------------------------
-- get/create dataset
--

nbTrainObjects = 50000
trainData = mnist_cluttered.createData({datasetPath='mnist/train.t7', megapatch_w=100, num_dist=8})

nbValidObjects = 10000
validData = mnist_cluttered.createData({datasetPath='mnist/valid.t7', megapatch_w=100, num_dist=8})

nbTestObjects = 10000
testData = mnist_cluttered.createData({datasetPath='mnist/test.t7', megapatch_w=100, num_dist=8})

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- settings taken from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua
-- using dampening = 0 seems to speed up convergence (optim.sgd's default is opt.momentum)
optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

inputs = torch.CudaTensor()
labels = torch.CudaTensor()

function getBatch(dataset, numExamples, geometry)
   local inputsCPU = torch.FloatTensor(numExamples,1,geometry[1],geometry[2])
   local labelsCPU = torch.FloatTensor(numExamples)

   for i = 1,numExamples do
      local observation, target = unpack(dataset.nextExample())
      inputsCPU[{i, 1, {}, {}}] = observation
      local _, label = target:max(1)
      labelsCPU[i] = label
   end

   return {inputsCPU, labelsCPU}
end

-- training function
function train(dataset, numObjects)
   -- local vars
   local time = sys.clock()

   -- do one epoch
   local numBatches = math.ceil(numObjects/opt.batchSize)
   for i = 1,numBatches do
      local batchSize = opt.batchSize
      if i == numBatches then
         batchSize = numObjects - (i-1) * opt.batchSize
      end
      local inputsCPU, labelsCPU = unpack(getBatch(dataset, batchSize, geometry))

      -- transfer over to GPU
      inputs:resize(inputsCPU:size()):copy(inputsCPU)
      labels:resize(labelsCPU:size()):copy(labelsCPU)

      local err, outputs
      feval = function(x)
         model:zeroGradParameters()
         outputs = model:forward(inputs)
         err = criterion:forward(outputs, labels)
         local gradOutputs = criterion:backward(outputs, labels)
         model:backward(inputs, gradOutputs)
         return err, gradParameters
      end
      if opt.optimizer == 'SGD' then
         optim.sgd(feval, parameters, optimState)
      elseif opt.optimizer == 'ADAM' then
         optim.adam(feval, parameters, optimState)
      else
         error('Unknown optimizer '..opt.optimizer)
      end

      local outputsCPU = outputs:float()
      confusion:batchAdd(outputsCPU, labelsCPU)
   end

   -- time taken
   time = sys.clock() - time
   local speed = numObjects / time

   -- print confusion matrix
   confusion:updateValids()
   print('train accuracy '..(confusion.totalValid * 100)..'%, speed '..speed..' examples/s')
   confusion:zero()

   -- save/log current net
   -- local filename = paths.concat(opt.save, 'mnist.net')
   -- os.execute('mkdir -p ' .. opt.save)
   -- print('saving network to '..filename)
   -- torch.save(filename, model)
end

-- test function
function test(dataset, numObjects, setName)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   local numBatches = math.ceil(numObjects/opt.batchSize)
   for i = 1,numBatches do
      local batchSize = opt.batchSize
      if i == numBatches then
         batchSize = numObjects - (i-1) * opt.batchSize
      end
      -- create mini batch
      local inputsCPU, labelsCPU = unpack(getBatch(dataset, batchSize, geometry))

      -- transfer over to GPU
      inputs:resize(inputsCPU:size()):copy(inputsCPU)
      labels:resize(labelsCPU:size()):copy(labelsCPU)

      -- test samples
      local preds = model:forward(inputs)
      local predsCPU = preds:float()

      -- confusion:
      confusion:batchAdd(predsCPU, labelsCPU)
   end

   -- timing
   time = sys.clock() - time
   local speed = numObjects / time

   confusion:updateValids()
   print(setName..' accuracy '..(confusion.totalValid * 100)..'%, speed '..speed..' examples/s')
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
for epoch = 1,opt.numEpochs do
   if opt.fixedTransformations then
      torch.manualSeed(opt.manualSeed)
   end

   print('Epoch '..epoch..'/'..opt.numEpochs)
   print('Learning rate '..opt.learningRate)

   -- train/test
   train(trainData, nbTrainObjects)
   test(validData, nbValidObjects, 'validation')
   test(testData, nbTestObjects, 'test')

   print('')

   -- decrease learning rate
   if opt.decayEpochs and epoch % opt.decayEpochs == 0 then
      opt.learningRate = opt.learningRate / 10
   end
end
