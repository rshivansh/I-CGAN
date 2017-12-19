
require 'image'
require 'nn'
optnet = require 'optnet'
disp = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

assert(loadfile("cfg/testConfig.lua"))(1)
torch.manualSeed(123)

local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end


-- Load net
local Anet = torch.load(opt.net)

-- Load samples for evaluating
print('Loading test dataset...')
local data = torch.load(opt.testSetPath)
local X = data.X
if X:min() >= 0 then 
    print('Seems images X are not in range [-1,1].\nAssuming range [0,1] and scaling to [-1,1].')
    X = X:add(1):div(2) -- Convert [-1,1] to [0,1]
end 
print(('Done. Loaded %.2f GB (%d images).'):format((4*X:size(1)*X:size(2)*X:size(3)*X:size(4))/2^30, X:size(1)))
local Y = data.Y:add(1):div(2) -- Convert [-1,1] to [0,1]
local nSamples = X:size(1)

-- Initialize batches
local batchX = torch.Tensor(opt.batchSize, X:size(2), X:size(3), X:size(4))

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    batchX = batchX:cuda()
    cudnn.convert(Anet, cudnn)
    Anet:cuda()
else
    Anet:float()
end
Anet:evaluate()

local function applyThreshold(y, dataset)
  -- Adapted for celebA
  for i=1,y:size(1) do
      for j=1,y:size(2) do
          local val = y[{{i},{j}}][1][1]
          if val > 0 then
              y[{{i},{j}}] = 1
          else
              y[{{i},{j}}] = 0
          end
      end
  end

  return y:int()
end

local function updateCM(CM, Ypred, Yreal)
    -- Ypred and Yreal are both {0, 1}
    local batchSz = Ypred:size(1)
    for i = 1, batchSz do
        for j = 1, Ypred:size(2) do
            -- Update jth CM
            local r = Yreal[i][j]+1
            local p = Ypred[i][j]+1
            CM[{{j},{r},{p}}] = CM[{{j},{r},{p}}] + 1
        end
    end
    
    return CM
end

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
batchX:copy(X[{{1,opt.batchSize},{},{},{}}])
optnet.optimizeMemory(Anet, batchX)

local CM = torch.IntTensor(Y:size(2), 2, 2):zero() -- There are Y:size(1) confusion matrices, where rows: real samples, cols: predicted samples
for batch = 1, nSamples-opt.batchSize+1, opt.batchSize  do
    -- Assign batch
    batchX:copy(X[{{batch,batch+opt.batchSize-1},{},{},{}}])
    local Yreal = Y[{{batch,batch+opt.batchSize-1},{}}]
    
    -- Predict attributes Y
    local Ypred = Anet:forward(batchX):float()
    
    -- Threshold Y
    Ypred = applyThreshold(Ypred, opt.dataset)
    
    CM = updateCM(CM, Ypred, Yreal)
    
    print(('%4d / %4d'):format(((batch-1) / opt.batchSize), math.ceil(nSamples / opt.batchSize)))
end

-- Print metrics for each CM
local accuracies = torch.FloatTensor(CM:size(1))
local precisions = accuracies:clone()
local recalls = accuracies:clone()
local f1scores = accuracies:clone()
local str_list = {'bald', 'bangs', 'black hair', 'blond', 'brown', 'eyebrows', 'eyeglasses', 'gray', 'makeup', 'male', 'mouth open', 'mustache', 'pale skin', 'receding hairline', 'smiling', 'straight hair', 'wavy hair', 'hat'}
for i=1, CM:size(1) do 
    local TN = CM[i][1][1]
    local FP = CM[i][1][2]
    local FN = CM[i][2][1]
    local TP = CM[i][2][2]
    precisions[i] = 100*(TP/(TP+FP))
    recalls[i] = 100*(TP/(TP+FN))
    -- Filter NaN 
    if precisions[i] ~= precisions[i] then precisions[i] = 100 end
    if recalls[i] ~= recalls[i] then recalls[i] = 100 end
    
    accuracies[i] = 100*((TP+TN)/(TP+TN+FP+FN))
    f1scores[i] = 2*((precisions[i]*recalls[i]) / (precisions[i]+recalls[i]))
    print(('%s\t\t Accuracy: %.2f%%\tF1Score: %.2f%%'):format(str_list[i], accuracies[i],f1scores[i]))
end

-- Filter NaN
--precisions[precisions:ne(precisions)] = 1
--recalls[recalls:ne(recalls)] = 1

print(('Mean accuracy: %.2f%%\tMean F1Score: %.2f%%'):format(torch.mean(accuracies), torch.mean(f1scores)))
