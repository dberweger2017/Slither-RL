"""
Spatial Hash Grid for efficient proximity queries.

Divides the world into cells of a fixed size. Entities are inserted
into the cell(s) they overlap, and queries only check nearby cells
instead of iterating over all entities.
"""

class SpatialHash:
    __slots__ = ('cell_size', 'inv_cell', 'grid')
    
    def __init__(self, cell_size=200):
        self.cell_size = cell_size
        self.inv_cell = 1.0 / cell_size
        self.grid = {}
    
    def clear(self):
        self.grid.clear()
    
    def _key(self, x, y):
        return (int(x * self.inv_cell), int(y * self.inv_cell))
    
    def insert(self, obj, x, y):
        """Insert an object at position (x, y)."""
        k = self._key(x, y)
        bucket = self.grid.get(k)
        if bucket is None:
            self.grid[k] = [obj]
        else:
            bucket.append(obj)
    
    def insert_rect(self, obj, x, y, radius):
        """Insert an object covering a bounding box around (x, y) ± radius."""
        min_cx = int((x - radius) * self.inv_cell)
        max_cx = int((x + radius) * self.inv_cell)
        min_cy = int((y - radius) * self.inv_cell)
        max_cy = int((y + radius) * self.inv_cell)
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                k = (cx, cy)
                bucket = self.grid.get(k)
                if bucket is None:
                    self.grid[k] = [obj]
                else:
                    bucket.append(obj)
    
    def query(self, x, y, radius):
        """Return all objects in cells overlapping the query circle."""
        min_cx = int((x - radius) * self.inv_cell)
        max_cx = int((x + radius) * self.inv_cell)
        min_cy = int((y - radius) * self.inv_cell)
        max_cy = int((y + radius) * self.inv_cell)
        result = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                bucket = self.grid.get((cx, cy))
                if bucket:
                    result.extend(bucket)
        return result
    
    def query_unique(self, x, y, radius):
        """Like query() but deduplicates (for objects spanning multiple cells)."""
        seen = set()
        min_cx = int((x - radius) * self.inv_cell)
        max_cx = int((x + radius) * self.inv_cell)
        min_cy = int((y - radius) * self.inv_cell)
        max_cy = int((y + radius) * self.inv_cell)
        result = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                bucket = self.grid.get((cx, cy))
                if bucket:
                    for obj in bucket:
                        obj_id = id(obj)
                        if obj_id not in seen:
                            seen.add(obj_id)
                            result.append(obj)
        return result
