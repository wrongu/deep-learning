--[[ train.lua

A straightforward implementation training a model that defaults to mnist and Goodfellow et al's Maxout model
]]--

torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Simple, straightforward neural network training with SGD')
cmd:text()
cmd:text('Options')
cmd:option('-seed', 404, 'random seed')
cmd:option('-minibatch', 128, 'minibatch size')
cmd:option('-epochs', 1000, 'number of training epochs (i.e. number of batches)')
cmd:option('-lr', 0.05, 'initial learning rate')
cmd:option('-momentum', 0.5, 'SGD momentum')
cmd:option('-dataset', 'datasets/mnist', 'name of data-loading module. see the README in the datasets/ directory regarding the expected format')
cmd:option('-model', 'models/mnist_maxoutnet', 'name of model-creating module')
cmd:option('-save_dir', 'models/trained/', 'directory to save .t7 models during training')
cmd:option('-save_every', 10, 'during training, save model every <save_every> epochs (note: final trained model will be saved regardless)')
cmd:option('-resume', nil, 'path to a .t7 file from which training should start (if nil, starts from scratch)')
cmd:text()

opts = cmd:parse(args)

function print_test_error(N)
	N = N or 10000
	local ncorrect = 0
	local log_avg_confidence = 0
	local idxs = torch.randperm(test_labels:size(1))
	for i=1,N do
		local idx = (idxs[i]-1) % test_labels:size(1) + 1
		net:forward(get_input(test_images, idx))
		local confidence, class = net.output:max(1)
		log_avg_confidence = log_avg_confidence + math.log(confidence[1])
		if class[1]-1 == test_labels[idx] then
			ncorrect = ncorrect + 1
		end
	end
	pcorrect = 100 * ncorrect / N
	log_avg_confidence = log_avg_confidence / N
	print(pcorrect .. "% correct, avg confidence " .. math.exp(log_avg_confidence))
end

function indicator(n,i,v)
	v = v or torch.zeros(n):float()
	v[i] = 1
	return v
end

function get_input(imgs, idx)
	return imgs[idx]:resize(1, imgs:size(2), imgs:size(3))
end

local criterion = nn.ClassNLLCriterion():float()

torch.manualSeed(123)

-- local grad = torch.Tensor()
local vout = indicator(10,1)
for epoch=1,num_epochs do
	net:zeroGradParameters()
	for img=1,minibatch_size do
		-- pick a training image at random
		local idx = math.floor(torch.uniform(1,train_labels:size(1)+1))
		local input = get_input(train_images, idx)
		local target = indicator(10, train_labels[idx]+1, vout)
		-- forward and back-propagation
		net:forward(input)
		criterion:forward(net.output, target)
		net:backward(input, criterion:backward(net.output, target))
	end
	-- minibatch update has been accumulated
	net:updateParameters(learning_rate)
	net:evaluate()
	print_test_error(500)
	net:training()
end