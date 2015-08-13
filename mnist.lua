--[[
mnist.lua

written by Richard Lange

provides MNIST training and testing data
]]--

require 'torch'

local function data_as_num(sub)
	local num = 0
	for c in string.gfind(sub,".") do
		num = num * 256 + string.byte(c)
	end
	return num
end

function load_mnist_images(file)
	-- do we need to do anything special here to manage GPU memory??
	local f = assert(io.open(file, "rb"))

	-- check data valid (first 4 bytes should make the number 2051 according to the specs)
	assert(data_as_num(f:read(4)) == 2051)

	local n_images = data_as_num(f:read(4))
	local rows = data_as_num(f:read(4))
	local cols = data_as_num(f:read(4))
	
	print(n_images .. " images at " .. rows .. "x" .. cols)

	local imagedata = torch.ByteStorage():string(f:read("*all"))
	local images = torch.ByteTensor():set(imagedata):resize(n_images, rows, cols)

	f:close()

	return images:int():float()
end

function load_mnist_labels(file)
	local f = assert(io.open(file, "rb"))

	-- check data valid (first 4 bytes should make the number 2049 according to the specs)
	assert(data_as_num(f:read(4)) == 2049)

	local n_labels = data_as_num(f:read(4))

	print(n_labels .. " labels")

	local labeldata = torch.ByteStorage():string(f:read("*all"))
	local labels = torch.ByteTensor():set(labeldata)

	f:close()

	return labels:int()
end