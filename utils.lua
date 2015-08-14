--[[ utils.lua: An assortment of helper functions

written by Richard Lange
]]--

-- zip_tables returns a new table whose ith index yields {tbl1[_min+i-1], tbl2[_min+i-1]}
-- In other words, this essentially combines the functionality of zip() with getting a subrange
function zip_tables(tbl1, tbl2, _min, _max)
	if #tbl1 ~= #tbl2 then
		print("WARNING cannot zip tables of different sizes")
		return nil
	end
	return setmetatable({}, {
		__index = function(t,k)
			local true_idx = k+_min-1
			if true_idx <= _max then
				return {tbl1[true_idx], tbl2[true_idx]}
			else
				return nil
			end
		end,
		__len = function() return _max-_min+1 end
	})
end