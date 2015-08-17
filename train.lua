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
cmd:option('-gpu', false, 'whether to use GPU (a boolean - no device management capabilities yet)')
cmd:option('-minibatch', 128, 'minibatch size')
cmd:option('-epochs', 1000, 'number of training epochs (i.e. number of batches)')
cmd:option('-lr', 0.05, 'initial learning rate')
cmd:option('-momentum', 0.5, 'SGD momentum')
cmd:option('-dataset', 'datasets.mnist', 'name of data-loading module. see the README in the datasets/ directory regarding the expected format')
cmd:option('-model', 'models.mnist_maxoutnet', 'name of model-creating module')
-- TODO extra model params here
cmd:option('-crit', 'nn.ClassNLLCriterion', 'an nn.Criterion class that is effectively appended to the model\'s output')
cmd:option('-save_dir', 'models/trained/', 'directory to save .net models during training')
cmd:option('-save_every', 10, 'during training, save model every <save_every> epochs (note: final trained model will be saved regardless)')
cmd:option('-resume', 0, 'number of epoch from which to restart (looks in save_dir for a <model name>.i<epoch>.net file)')
cmd:option('-liveeval', 500, 'number of test examples to (randomly) evaluate and print per epoch')
cmd:text()

opts = cmd:parse(arg)

function output_is_equal(a, b, tensorL2tolerance)
	tensorL2tolerance = tensorL2tolerance or 0
	if type(a) == type(b) then
		if torch.isTensor(a) and torch.isTensor(b) then
			if a:isSameSizeAs(b) then
				return (a - b):norm() <= tensorL2tolerance
			else
				return false
			end
		else
			return a == b
		end
	end
	return false
end

function evaluate(model, testset, N)
	N = N or 500
	local ncorrect = 0
	local log_avg_confidence = 0
	local idxs = torch.randperm(#testset)
	for i=1,N do
		-- modulo index in case N > #testset
		local idx = (idxs[i]-1) % #testset + 1
		-- do classification
		local correct = testset[idx][2]
		local clazz = model:forward(testset[idx][1])
		-- get class (index of max value in last layer).
		-- Output is flattened with view() so max(1) is agnostic to the model
		local confidence, class = clazz:view(clazz:numel()):max(1)

		log_avg_confidence = log_avg_confidence + confidence[1]

		if output_is_equal(class:float(), correct:float()) then
			ncorrect = ncorrect + 1
		end
	end
	pcorrect = 100 * ncorrect / N
	log_avg_confidence = log_avg_confidence / N -- end result is geometric mean of probabilities
	print(pcorrect .. "% correct, avg confidence " .. math.exp(log_avg_confidence))
end

function get_save_filename(itr)
	local fname = opts.model .. '.i' .. itr .. '.t7'
	return paths.concat(opts.save_dir, fname)
end

-- set up according to options
torch.manualSeed(opts.seed)

local data = require(opts.dataset)
local model = require(opts.model):float()
local criterion = torch.factory(opts.crit)()
-- factory doesn't call __init() so we do it manually
criterion:__init()
criterion:float()

-- load model from save file if requested
if opts.resume > 0 then
	if opts.resume == opts.epochs then
		opts.resume = "final"
	end
	local fname = get_save_filename(opts.resume)
	local try_load = torch.load(fname)
	if try_load then
		model = try_load
		print("Loaded from " .. fname)
	else
		print("WARNING: could not load from " .. fname .. "! Starting a blank model instead")
	end
end

if opts.gpu then
	model:cuda()
	criterion:cuda()
	-- TODO batch-handling of data transfers to the GPU?
end

-- pre-run evaluation
evaluate(model, data['test'], opts.liveeval)

-- permute training data
local indices = torch.randperm(#data['train'])
local i = 1 -- loops in order 1..#data across minibatches

for epoch=opts.resume+1,opts.epochs do
	model:zeroGradParameters()
	for example=1,opts.minibatch do
		-- pick a training image at random
		local input = data['train'][indices[i]][1]
		local target = data['train'][indices[i]][2]

		-- forward and back-propagation
		local model_out = model:forward(input)
		criterion:forward(model_out, target)
		model:backward(input, criterion:backward(model_out, target))

		-- move on to next example
		i = (i % #data['train']) + 1
	end
	-- minibatch update has been accumulated
	-- (TODO) include momentum
	model:updateParameters(opts.lr / opts.minibatch)
	model:evaluate()
	evaluate(model, data['validate'], opts.liveeval)
	model:training()

	-- save every so often
	if epoch % opts.save_every == 0 then
		local fname = get_save_filename(epoch)
		torch.save(fname, model)
		print("Saved " .. fname)
	end
end

-- save final result
torch.save(get_save_filename('final'), model)