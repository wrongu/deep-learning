--[[ nnxx.lua

Access to my own custom Modules via "require 'nnxx'"

So-named because the 'extensions' package nnx is already taken

]]--

require 'nn'

-- note that later modules may depend on earlier ones, so the order here matters
dofile('nnxx/Unfold.lua')
dofile('nnxx/Maxout.lua')