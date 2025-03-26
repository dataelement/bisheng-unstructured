local function allrows (tbl)
  local rows = pandoc.List{}
  rows:extend(tbl.head.rows)
  tbl.bodies:map(function (body) rows:extend(body.body) end)
  rows:extend(tbl.foot.rows)
  return rows
end

local function nested_rows (cell)
  local tbl = cell.contents[1]
  if tbl and tbl.t == 'Table' then
    return allrows(tbl)
  end
  return nil
end

function Table (tbl)
  local newhead = pandoc.TableHead()
  for i, row in ipairs(tbl.head.rows) do
    for j, cell in ipairs(row.cells) do
      local currow = row:clone()
      local nested_rows = nested_rows(cell)
      if nested_rows and #nested_rows > 0 then
        for jj=1, j-1 do
          currow.cells[jj].row_span = row.cells[jj].row_span + #nested_rows - 1
        end
        local firstnested = table.remove(nested_rows, 1)
        -- assume the nested table only has a single row
        currow.cells[j] = firstnested.cells[1]
        newhead.rows:insert(currow)
        newhead.rows:extend(nested_rows)
        goto continue
      end
    end
    newhead.rows:insert(row)
    ::continue::
  end
  tbl.head = newhead
  return tbl
end
