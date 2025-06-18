class Port:
    def __init__(self, block, offset, is_input):
        # Store reference to the parent block
        self.block = block
        # Store the offset from the block's top-left corner
        self.offset = offset
        # Flag indicating if this is an input port (True) or output port (False)
        self.is_input = is_input
        # List to store all connections made to/from this port
        self.connections = []

    @property
    def position(self):
        # Calculate the absolute position of the port on the screen
        # by adding the block's position to the port's offset
        bx, by = self.block.rect.topleft
        ox, oy = self.offset
        return (bx + ox, by + oy)
