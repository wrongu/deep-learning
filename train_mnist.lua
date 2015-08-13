--[[ train_mnist.lua

A straightforward implementation training a model to recognize MNIST digits
]]--

local net = require 'mnist_maxoutnet'
require 'nn'
require 'mnist'

minibatch_size = 128
num_epochs = 1000
learning_rate = 0.05

train_images = load_mnist_images('../datasets/mnist/train-images-idx3-ubyte')
train_labels = load_mnist_labels('../datasets/mnist/train-labels-idx1-ubyte')
test_images = load_mnist_images('../datasets/mnist/t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('../datasets/mnist/t10k-labels-idx1-ubyte')

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